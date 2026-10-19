"""
난독화 한글 리뷰 복원 - 4단계: 노이즈 채널(01) + 태거(02) 앙상블

01 의 빔 서치 점수에 태거의 글자 확률을 하나 더 더한다.
    score(y) = 채널 log P(x|y) + lm_weight * log P_LM(y) + dict_weight * P_사전(y)
               + tagger_weight * (log P_태거(초) + log P_태거(중) + log P_태거(종))
- 01 은 받침·연음·사전에 강하고, 태거는 넓은 문맥(뉴스/누수 같은 의미 판단)에 강하다.
- 태거 1순위 글자가 01 후보에 없으면 후보에 추가한다.
- 태거는 재학습 없이 02 체크포인트(tagger_best.pt)를 쓰고, 추론은 한 번만 한 뒤 조합별로 빔 서치만 다시 돈다.

사용 예 (Colab, GPU 권장 — 태거 추론. 빔 서치는 CPU):
    !python ensemble.py --mode val      # tagger_weight 등 탐색 -> outputs/best_ensemble.json
    !python ensemble.py --mode test     # best 설정으로 제출 파일
"""
import argparse
import itertools
import json
import os
import time

import pandas as pd
import torch

import noisy_channel as nc
from neural_tagger import CharVocab, build_masks, predict_logprobs, split_train
from tagger_dict import load_tagger

ENS_KEYS = nc.PARAM_KEYS + ["tagger_weight"]
DEFAULT_GRID = "tagger_weight=0,0.3,0.6,1.0,1.5;lm_weight=0.9,1.1;dict_oov=0.25,0.5"


def parse_grid(spec, base):
    grid = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        k, vs = (t.strip() for t in part.split("=", 1))
        if k not in ENS_KEYS:
            raise SystemExit(f"탐색할 수 없는 파라미터: {k} (가능: {ENS_KEYS})")
        cast = type(base[k])
        grid[k] = ([v.strip().lower() in ("1", "true", "yes") for v in vs.split(",")] if cast is bool
                   else [cast(v) for v in vs.split(",")])
    return grid


def base_params(args):
    """01 의 best_params.json 을 출발점으로 (없으면 noisy_channel 기본값)."""
    p = {"order": 5, "cand_thr": 0.005, "no_pair": False, "dict_min_len": 2, "dict_oov": 1.0,
         "beam": 8, "topk": 12, "lm_weight": 1.3, "dict_weight": 3.0,
         "aug_file": "", "aug_lm_n": 0, "aug_dict_weight": 0.0, "aug_min_count": 2}
    src = "noisy_channel 기본값"
    path = os.path.join(args.out_dir, "best_params.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            best = json.load(f)
        p.update(best["params"])
        src = f"best_params.json (01 val F1 {best['f1']:.4f})"
    p["tagger_weight"] = 0.0
    return p, src


def tagger_lps(args, ckpt, texts, rows_df, device):
    """태거 위치별 log-prob. 마스크는 체크포인트 학습 때와 같은 채널 기준."""
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    vocab = CharVocab(list(full.input) + list(full.output) + list(test.input))
    model, a, epoch = load_tagger(ckpt, vocab, device)
    ch = nc.JamoChannel(nc.aligned_rows(rows_df), cand_thr=a.cand_thr)
    t = time.time()
    lps = predict_logprobs(model, vocab, build_masks(ch), list(texts), device, win=a.max_len)
    print(f"  tagger {os.path.basename(ckpt)} (epoch {epoch}) inference {time.time() - t:.1f}s")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return lps, a, epoch


def log_row(out_dir, row):
    path = os.path.join(out_dir, "ens_runs.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run_val(args, device):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    ckpt = args.ckpt or os.path.join(args.out_dir, "tagger_best.pt")
    base, src = base_params(args)
    print(f"  출발 파라미터 [{src}]: {base}")

    # 태거 체크포인트와 같은 분할 (noisy_channel.split_val 과도 동일: seed 42, 10%)
    state_args = torch.load(ckpt, map_location="cpu")["args"]
    if state_args.get("full"):
        raise SystemExit("val 모드에는 val 을 빼고 학습한 체크포인트(tagger_best.pt)가 필요합니다.")
    tr, va = split_train(full, state_args["val_ratio"], state_args["seed"])
    if args.val_limit:
        va = va.iloc[: args.val_limit]
    gts = list(va.output)
    lps, _, epoch = tagger_lps(args, ckpt, va.input, tr, device)

    grid = parse_grid(args.grid, base)
    combos = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
    combos.sort(key=lambda c: tuple(str(c.get(k, base[k])) for k in nc.BUILD_KEYS))
    print(f"  {len(combos)} 조합 x 검증 {len(va)}행")
    built, decoder = None, None
    best = (-1.0, None, None)
    for i, combo in enumerate(combos, 1):
        p = {**base, **combo}
        a = argparse.Namespace(**p)
        bk = tuple(p[k] for k in nc.BUILD_KEYS)
        if bk != built:
            decoder, _ = nc.build(tr, a)
            built = bk
        decoder.beam, decoder.topk = a.beam, a.topk
        decoder.lm_weight, decoder.dict_weight, decoder.tagger_weight = a.lm_weight, a.dict_weight, a.tagger_weight
        decoder.cand_cache.clear()
        t = time.time()
        preds = nc.decode_all(decoder, list(va.input), args.workers, lps=lps)
        f1 = nc.mean_f1(preds, gts)
        log_row(args.out_dir, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ckpt": os.path.basename(ckpt),
                               "epoch": epoch, "n_val": len(va), **p, "f1": round(f1, 5),
                               "sec": round(time.time() - t, 1)})
        mark = ""
        if f1 > best[0]:
            best = (f1, p, preds)
            mark = " *"
        print(f"  [{i}/{len(combos)}] {combo} -> F1 {f1:.4f}{mark}", flush=True)
        # 중간에 끊겨도 best 가 남도록 매번 저장
        with open(os.path.join(args.out_dir, "best_ensemble.json"), "w", encoding="utf-8") as f:
            json.dump({"params": best[1], "f1": best[0], "n_val": len(va), "ckpt": os.path.basename(ckpt),
                       "epoch": epoch, "done": f"{i}/{len(combos)}",
                       "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=1)

    f1, p, preds = best
    print(f"best F1 {f1:.4f} {p}")
    va.assign(pred=preds, f1=[nc.f1_score(x, g) for x, g in zip(preds, gts)]).to_csv(
        os.path.join(args.out_dir, "val_pred_ensemble.csv"), index=False, encoding="utf-8-sig"
    )


def run_test(args, device):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    ckpt = args.ckpt or os.path.join(args.out_dir, "tagger_best.pt")
    path = os.path.join(args.out_dir, "best_ensemble.json")
    if not os.path.exists(path):
        raise SystemExit("outputs/best_ensemble.json 이 없습니다. 먼저 --mode val 을 실행하세요.")
    with open(path, encoding="utf-8") as f:
        best = json.load(f)
    p = best["params"]
    print(f"  사용 파라미터 [best_ensemble.json, val F1 {best['f1']:.4f}]: {p}")
    lps, _, epoch = tagger_lps(args, ckpt, test.input, full, device)
    decoder, _ = nc.build(full, argparse.Namespace(**p))
    decoder.tagger_weight = p["tagger_weight"]
    preds = nc.decode_all(decoder, list(test.input), args.workers, lps=lps)
    sub["output"] = sub.ID.map(dict(zip(test.ID, preds)))
    out = os.path.join(args.out_dir, "submission_ensemble.csv")
    sub.to_csv(out, index=False, encoding="utf-8-sig")
    with open(os.path.join(args.out_dir, "submission_ensemble_params.json"), "w", encoding="utf-8") as f:
        json.dump({"params": p, "val_f1": best["f1"], "ckpt": ckpt, "epoch": epoch,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=1)
    print(f"  saved {out} (+ _params.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "test"], default="val")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--ckpt", default="", help="태거 체크포인트 (기본: outputs/tagger_best.pt)")
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--grid", default=DEFAULT_GRID,
                    help="탐색 범위. 나머지 값은 best_params.json(01) 에서 가져옴. 예: 'tagger_weight=0.5,1;beam=16,24'")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(vars(args), "| device", device)
    (run_val if args.mode == "val" else run_test)(args, device)


if __name__ == "__main__":
    main()
