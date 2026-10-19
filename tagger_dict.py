"""
난독화 한글 리뷰 복원 - 3단계: 신경망 태거 + 어절 사전 결합 디코딩 (재학습 없음)

02 에서 학습한 태거 체크포인트를 불러와, 글자마다
    score(y) = log P_tagger(초성) + log P_tagger(중성) + log P_tagger(종성) + dict_weight * P_dict(y)
가 최대인 후보 음절을 고른다.
- P_dict : 01 의 어절 사전(WordDict). 입력 어절과 모음 골격이 같은 train 어절들을
           '난독화 확률(ㄱㄲㅋ 등 같은 그룹이면 높음) x 빈도' 로 비교한 사후확률을 글자별로 합친 것.
           사전에 없는 어절이면 'OOV' 확률이 커져서 보너스가 거의 0 → 태거 판단을 그대로 따름.
- val 모드 : dict_weight x dict_oov 조합을 검증하고 best 를 outputs/best_tagger_dict.json 에 저장
             (dict_weight 0 = 태거 단독 → 사전 효과를 바로 비교)
- test 모드 : best 설정으로 제출 파일 생성

사용 예 (Colab, GPU 권장):
    !python tagger_dict.py --mode val
    !python tagger_dict.py --mode test
"""
import argparse
import json
import math
import os
import time

import pandas as pd
import torch

from neural_tagger import CandidateDecoder, CharVocab, Tagger, build_masks, predict_logprobs, split_train
from noisy_channel import (
    JamoChannel, WordDict, aligned_rows, comp, dec, dict_report, f1_score, hangul_runs, is_hangul, mean_f1,
)


def load_tagger(path, vocab, device):
    state = torch.load(path, map_location=device)
    a = argparse.Namespace(**state["args"])
    model = Tagger(vocab.size, d=a.d, layers=a.layers, heads=a.heads, ff=a.d * 4,
                   drop=a.dropout, max_pos=max(512, a.max_len)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, a, state["epoch"]


def dict_bonus(wd, x):
    """위치 -> {원본 글자: 사전 사후확률}. (가중치는 디코딩 때 곱함)"""
    bonus = {}
    for i0, xr in hangul_runs(x):
        if len(xr) < wd.min_len:
            continue
        post = wd.posterior(xr)
        if not post:
            continue
        for j, d in enumerate(post):
            if d:
                bonus[i0 + j] = dict(d)
    return bonus


class DictCandidateDecoder(CandidateDecoder):
    """CandidateDecoder + 사전 보너스. 사전이 가리키는 글자가 후보에 없으면 추가."""

    def decode(self, x, lp, bonus=None, weight=0.0):
        la, lb, lg = lp
        out = list(x)
        for i, xc in enumerate(x):
            if not is_hangul(xc):
                continue
            cands = self.cands(xc)
            bo = bonus.get(i) if (bonus and weight > 0) else None
            if bo:
                have = {y for y, *_ in cands}
                cands = cands + [(y, *dec(y)) for y in bo if y not in have]
            best, bs = None, -1e18
            for y, a, b, g in cands:
                s = la[i, a] + lb[i, b] + lg[i, g]
                if bo:
                    s += weight * bo.get(y, 0.0)
                if s > bs:
                    best, bs = y, s
            out[i] = best or comp(int(la[i].argmax()), int(lb[i].argmax()), int(lg[i].argmax()))
        return "".join(out)


def log_row(out_dir, row):
    path = os.path.join(out_dir, "tagger_dict_runs.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def floats(s):
    return [float(v) for v in s.split(",") if v.strip()]


def run_val(args, device):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    vocab = CharVocab(list(full.input) + list(full.output) + list(test.input))
    ckpt = args.ckpt or os.path.join(args.out_dir, "tagger_best.pt")
    model, a, epoch = load_tagger(ckpt, vocab, device)
    if a.full:
        raise SystemExit("val 모드에는 val 을 빼고 학습한 체크포인트(tagger_best.pt)가 필요합니다.")
    print(f"loaded {ckpt} (epoch {epoch})")

    # 체크포인트 학습 때와 같은 분할·채널
    tr, va = split_train(full, a.val_ratio, a.seed)
    if args.val_limit:
        va = va.iloc[: args.val_limit]
    gts = list(va.output)
    ch = JamoChannel(aligned_rows(tr), cand_thr=a.cand_thr)
    masks = build_masks(ch)
    vocab_chars = {c for t in tr.output for c in t if is_hangul(c)}
    dec_ = DictCandidateDecoder(ch, vocab_chars)
    wd = WordDict(tr.output, ch, vocab_chars, min_len=args.dict_min_len)
    dict_report(wd, list(va.input), gts)

    t = time.time()
    lps = predict_logprobs(model, vocab, masks, list(va.input), device, win=a.max_len)
    print(f"  tagger inference {time.time() - t:.1f}s")

    weights, oovs = floats(args.weights), floats(args.oovs)
    best = (-1.0, None, None)
    for oov in oovs:
        wd.log_oov, wd.cache = math.log(oov), {}
        bonuses = [dict_bonus(wd, x) for x in va.input]
        for w in weights:
            if w == 0 and oov != oovs[0]:
                continue  # 사전 끔은 oov 와 무관 → 한 번만
            preds = [dec_.decode(x, lp, bo, w) for x, lp, bo in zip(va.input, lps, bonuses)]
            f1 = mean_f1(preds, gts)
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ckpt": os.path.basename(ckpt), "epoch": epoch,
                   "n_val": len(va), "dict_weight": w, "dict_oov": oov, "dict_min_len": args.dict_min_len,
                   "f1": round(f1, 5)}
            log_row(args.out_dir, row)
            mark = ""
            if f1 > best[0]:
                best = (f1, {"dict_weight": w, "dict_oov": oov, "dict_min_len": args.dict_min_len}, preds)
                mark = " *"
            tag = "tagger only" if w == 0 else f"dict_weight {w:<4} dict_oov {oov}"
            print(f"  {tag:<34} F1 {f1:.4f}{mark}", flush=True)

    f1, params, preds = best
    print(f"best F1 {f1:.4f} {params}")
    va.assign(pred=preds, f1=[f1_score(p, g) for p, g in zip(preds, gts)]).to_csv(
        os.path.join(args.out_dir, "val_pred_tagger_dict.csv"), index=False, encoding="utf-8-sig"
    )
    with open(os.path.join(args.out_dir, "best_tagger_dict.json"), "w", encoding="utf-8") as fp:
        json.dump({"params": params, "f1": f1, "n_val": len(va), "ckpt": os.path.basename(ckpt), "epoch": epoch,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, ensure_ascii=False, indent=1)


def run_test(args, device):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    vocab = CharVocab(list(full.input) + list(full.output) + list(test.input))
    ckpt = args.ckpt or os.path.join(args.out_dir, "tagger_best.pt")
    model, a, epoch = load_tagger(ckpt, vocab, device)

    params = {"dict_weight": floats(args.weights)[-1], "dict_oov": floats(args.oovs)[0],
              "dict_min_len": args.dict_min_len}
    source = "명령줄 값 (--weights 마지막 값, --oovs 첫 값)"
    best_path = os.path.join(args.out_dir, "best_tagger_dict.json")
    if not args.ignore_best and os.path.exists(best_path):
        with open(best_path, encoding="utf-8") as fp:
            best = json.load(fp)
        params = best["params"]
        source = f"best_tagger_dict.json (val F1 {best['f1']:.4f})"
    print(f"loaded {ckpt} (epoch {epoch}) | 사용 파라미터 [{source}]: {params}")

    ch = JamoChannel(aligned_rows(full), cand_thr=a.cand_thr)
    masks = build_masks(ch)
    vocab_chars = {c for t in full.output for c in t if is_hangul(c)}
    dec_ = DictCandidateDecoder(ch, vocab_chars)
    wd = WordDict(full.output, ch, vocab_chars, min_len=params["dict_min_len"], oov_count=params["dict_oov"])
    lps = predict_logprobs(model, vocab, masks, list(test.input), device, win=a.max_len)
    preds = [dec_.decode(x, lp, dict_bonus(wd, x), params["dict_weight"]) for x, lp in zip(test.input, lps)]
    sub["output"] = sub.ID.map(dict(zip(test.ID, preds)))
    path = os.path.join(args.out_dir, "submission_tagger_dict.csv")
    sub.to_csv(path, index=False, encoding="utf-8-sig")
    with open(os.path.join(args.out_dir, "submission_tagger_dict_params.json"), "w", encoding="utf-8") as fp:
        json.dump({"params": params, "source": source, "ckpt": ckpt, "epoch": epoch,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, ensure_ascii=False, indent=1)
    print(f"saved {path} (+ _params.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "test"], default="val")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--ckpt", default="", help="태거 체크포인트 (기본: outputs/tagger_best.pt)")
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--weights", default="0,0.5,1,2,3,5", help="탐색할 dict_weight 목록 (0 = 태거 단독)")
    ap.add_argument("--oovs", default="0.5,1,2", help="탐색할 dict_oov 목록 (클수록 사전을 덜 믿음)")
    ap.add_argument("--dict_min_len", type=int, default=2)
    ap.add_argument("--ignore_best", action="store_true", help="test 때 best_tagger_dict.json 무시")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(vars(args), "| device", device)
    (run_val if args.mode == "val" else run_test)(args, device)


if __name__ == "__main__":
    main()
