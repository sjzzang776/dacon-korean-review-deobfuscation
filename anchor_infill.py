"""
난독화 한글 리뷰 복원 - 8단계: 확실한 어절 고정 + 나머지를 앞뒤 문맥·유사도로 추론 (공개 언어모델)

아이디어(사용자): 이전 최고 모델의 결과에서 확실한 단어를 먼저 고정하고, 나머지 단어를 정방향·역방향
문맥과 유사도를 함께 보며 하나씩 추론한다.

  1. 고정(anchor): 어절(한글 구간) 단위로 01 결과와 참조 모델(기본 v2 태거) 결과가 같으면 고정.
     - 검증 기준 01·v2 일치 어절은 91.4%, 그 정확도 98.2% (불일치 어절은 01 도 57% 만 맞음)
     - --anchor length --min_len 4 : 사용자 원안(글자 수 기준). 단, 검증에서 어절이 길수록
       01 정확도가 오히려 낮았음(2글자 96.0% → 5글자 91.4%) → 비교용으로 제공
  2. 후보: 불일치 어절마다 {01 결과, 참조 모델 결과, 01 이 그 위치에서 낸 상위 nbest 후보}
  3. 점수 = 유사도 + w × 문맥
     - 유사도: 01 채널의 log P(입력 어절 | 후보)  (그 후보가 이 입력 형태로 난독화될 확률)
     - 문맥: 공개 언어모델이 계산한 '후보를 끼운 앞뒤 ctx 글자 창' 전체의 log P → 앞·뒤 문맥 동시 반영
  4. 순서: 불일치 어절을 긴 것부터(--order length) 하나씩 확정, 확정된 결과는 다음 어절의 문맥이 됨.
     passes=2 이면 두 번째는 역방향(뒤→앞)으로 다시 훑으며 바뀐 문맥으로 재결정.

사용 예 (Colab, GPU):
    !python -u anchor_infill.py --mode val
    !python -u anchor_infill.py --mode val --anchor length --min_len 4    # 사용자 원안 비교
    !python -u anchor_infill.py --mode test
"""
import argparse
import collections
import json
import math
import os
import pickle
import re
import time
from multiprocessing import Pool

import pandas as pd

import noisy_channel as nc
from neural_tagger import split_train

# 참조 모델: (검증 예측 파일, test 예측 파일, test 파일이 없을 때 만드는 방법)
REFS = {
    "v2": ("val_pred_tagger_v2_scratch.csv", "submission_tagger_v2_scratch.csv",
           "05 노트북: !python -u tagger_v2.py --mode predict --run_name scratch --workers 6"),
    "stack": ("val_pred_stack.csv", "submission_stack.csv", "06 노트북: !python -u stacker.py --mode predict"),
    "v3": ("val_pred_tagger_v2_v3.csv", "submission_tagger_v2_v3.csv",
           "11 노트북: !python -u tagger_v2.py --mode predict --run_name v3 --workers 8"),
}


def nc_params(out_dir):
    p = {"order": 5, "cand_thr": 0.005, "no_pair": False, "dict_min_len": 2, "dict_oov": 0.5,
         "beam": 16, "topk": 12, "lm_weight": 1.1, "dict_weight": 3.0}
    path = os.path.join(out_dir, "best_params.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            p.update(json.load(f)["params"])
    return p


# --------------------------------------------------------------------------- #
# 1·2. 고정 / 후보 (CPU)
# --------------------------------------------------------------------------- #
_G = {}


def _sentence_items(job):
    """한 문장의 불일치 어절별 후보와 유사도."""
    x, p01, refs = job
    dec, args = _G["dec"], _G["args"]
    items, n_words, n_anchor = [], 0, 0
    for m in re.finditer(r"[가-힣]+", x):
        s, e = m.span()
        n_words += 1
        w01 = p01[s:e]
        agree = all(r[s:e] == w01 for r in refs)
        if args.anchor == "agree":
            anchored = agree
        elif args.anchor == "length":
            anchored = (e - s) >= args.min_len
        else:  # both: 일치하거나 충분히 긴 어절
            anchored = agree or (e - s) >= args.min_len
        # 09-A: 01·참조가 일치해도 그 답이 train 정답에 한 번도 없던 어절이면 고정하지 않는다
        #        (검증: 이런 고정 어절은 정확도 90.3%, train 에 10회 이상 나온 어절은 99.1%)
        if anchored and args.unanchor_oov and _G["train_words"].get(w01, 0) == 0:
            anchored = False
        if anchored:
            n_anchor += 1
            continue
        cands = [w01] + [r[s:e] for r in refs]
        if args.nbest > 0:
            cands += [y for y, _ in dec.decode_nbest(x[s:e], args.nbest, ctx_y=p01[:s], ctx_x=x[:s], final=False)]
        uniq = []
        for c in cands:
            if len(c) == e - s and all(nc.is_hangul(ch) for ch in c) and c not in uniq:
                uniq.append(c)
        sims = [dec.ch.seq_logp(x[s:e], c) for c in uniq]
        items.append({"s": s, "e": e, "cands": uniq, "sim": sims})
    return {"items": items, "n_words": n_words, "n_anchor": n_anchor}


def make_items(train_df, xs, p01s, refs_preds, args):
    p = nc_params(args.out_dir)
    if getattr(args, "aug_file", ""):  # 10단계: test 때는 train 전체 모델이 생성한 문장 파일로 교체
        p["aug_file"] = args.aug_file
    if args.cand_beam:  # 09-B: 후보 생성 때만 빔을 넓힘 (검증: 빔 64·후보 30 이면 누락 정답의 35% 가 후보에 들어옴)
        p["beam"] = args.cand_beam
    dec, _ = nc.build(train_df, argparse.Namespace(**p))
    train_words = collections.Counter(m.group() for y in train_df.output for m in re.finditer(r"[가-힣]+", str(y)))
    _G.update(dec=dec, args=args, train_words=train_words)
    jobs = [(x, p01, [r[i] for r in refs_preds]) for i, (x, p01) in enumerate(zip(xs, p01s))]
    t = time.time()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            res = list(pool.imap(_sentence_items, jobs, chunksize=8))
    else:
        res = [_sentence_items(j) for j in jobs]
    nw = sum(r["n_words"] for r in res)
    na = sum(r["n_anchor"] for r in res)
    nd = sum(len(r["items"]) for r in res)
    nc_ = sum(len(it["cands"]) for r in res for it in r["items"])
    print(f"  어절 {nw} | 고정 {na} ({na / max(nw, 1):.1%}) | 추론 대상 {nd} (평균 후보 {nc_ / max(nd, 1):.1f}개) "
          f"| {time.time() - t:.0f}s")
    return res


# --------------------------------------------------------------------------- #
# 3·4. 언어모델 문맥 + 순차 확정 (GPU)
# --------------------------------------------------------------------------- #
class LMScorer:
    """문자열 전체의 log P (BOS 다음부터 모든 토큰). 같은 문자열은 캐시."""

    def __init__(self, name, batch_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32).to(self.device).eval()
        self.start = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.batch_tokens, self.cache, self.n_forward = batch_tokens, {}, 0
        print(f"  LM {name} ({sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B) on {self.device}")

    def __call__(self, texts):
        torch = self.torch
        todo = sorted({t for t in texts if t not in self.cache}, key=len)
        enc = {t: [self.start] + self.tok(t, add_special_tokens=False).input_ids for t in todo}
        i = 0
        with torch.no_grad():
            while i < len(todo):
                L = len(enc[todo[min(i + 255, len(todo) - 1)]])
                bs = max(1, min(256, self.batch_tokens // max(L, 1)))
                chunk = todo[i : i + bs]
                n = max(len(enc[t]) for t in chunk)
                ids = torch.full((len(chunk), n), self.pad, dtype=torch.long)
                att = torch.zeros((len(chunk), n), dtype=torch.long)
                for r, t in enumerate(chunk):
                    ids[r, : len(enc[t])] = torch.tensor(enc[t])
                    att[r, : len(enc[t])] = 1
                ids, att = ids.to(self.device), att.to(self.device)
                logits = self.model(input_ids=ids, attention_mask=att).logits
                for r, t in enumerate(chunk):
                    m = len(enc[t])
                    if m < 2:
                        self.cache[t] = 0.0
                        continue
                    lp = torch.log_softmax(logits[r, : m - 1].float(), -1)
                    self.cache[t] = float(lp.gather(-1, ids[r, 1:m, None]).sum())
                del logits
                self.n_forward += len(chunk)
                i += bs
        return [self.cache[t] for t in texts]


def resolve(xs, p01s, items, lm, w, args, wf=0.0, margin=0.0, freq=None):
    """불일치 어절을 순서대로 확정. 같은 단계의 어절들은 문장끼리 묶어 한 번에 LM 계산.

    09 옵션 (공개 LM 이 리뷰 은어·구어를 표준어 쪽으로 바꾸는 실수 대책. 예: 지립니다 -> 질립니다)
      wf     : train 정답에 그 어절이 나온 횟수의 log(1+횟수) 가중치 -> 리뷰에서 실제 쓰이는 표현에 가점
      margin : 01 의 답(후보 0)을 다른 후보로 바꾸려면 점수가 이만큼 이상 높아야 함 -> 애매하면 01 유지
    """
    cur = [list(p) for p in p01s]
    orders = []
    for res in items:
        its = list(range(len(res["items"])))
        if args.order == "length":  # 사용자 원안: 긴 어절부터 (같으면 앞쪽부터)
            its.sort(key=lambda k: (-(res["items"][k]["e"] - res["items"][k]["s"]), res["items"][k]["s"]))
        orders.append(its)
    for ps in range(args.passes):
        seqs = [o if ps % 2 == 0 else list(reversed(o)) for o in orders]
        for step in range(max((len(o) for o in seqs), default=0)):
            texts, owners = [], []
            for si, seq in enumerate(seqs):
                if step >= len(seq):
                    continue
                it = items[si]["items"][seq[step]]
                s, e = it["s"], it["e"]
                lo, hi = max(0, s - args.ctx), min(len(cur[si]), e + args.ctx)
                left, right = "".join(cur[si][lo:s]), "".join(cur[si][e:hi])
                for k, c in enumerate(it["cands"]):
                    texts.append(left + c + right)
                    owners.append((si, seq[step], k))
            if not texts:
                continue
            scores = lm(texts)
            best, score01 = {}, {}
            for (si, ii, k), lp in zip(owners, scores):
                it = items[si]["items"][ii]
                sc = it["sim"][k] + w * lp
                if wf:
                    sc += wf * math.log1p(freq.get(it["cands"][k], 0))
                if k == 0:
                    score01[(si, ii)] = sc  # 후보 0 = 01 의 답
                if (si, ii) not in best or sc > best[(si, ii)][0]:
                    best[(si, ii)] = (sc, k)
            for (si, ii), (sc, k) in best.items():
                if margin and k != 0 and sc - score01[(si, ii)] < margin:
                    k = 0
                it = items[si]["items"][ii]
                cur[si][it["s"] : it["e"]] = list(it["cands"][k])
    return ["".join(c) for c in cur]


def oracle(p01s, items, gts):
    out = []
    for p, res, g in zip(p01s, items, gts):
        c = list(p)
        for it in res["items"]:
            seg = g[it["s"] : it["e"]]
            best = max(it["cands"], key=lambda y: sum(a == b for a, b in zip(y, seg)))
            c[it["s"] : it["e"]] = list(best)
        out.append("".join(c))
    return out


def anchor_report(xs, p01s, items, gts):
    """고정 어절의 정확도 / 추론 대상에서 01 이 맞은 비율."""
    a_n = a_ok = d_n = d_ok = 0
    for x, p, res, g in zip(xs, p01s, items, gts):
        disputed = {(it["s"], it["e"]) for it in res["items"]}
        for m in re.finditer(r"[가-힣]+", x):
            s, e = m.span()
            ok = p[s:e] == g[s:e]
            if (s, e) in disputed:
                d_n += 1
                d_ok += ok
            else:
                a_n += 1
                a_ok += ok
    print(f"  고정 어절 정확도 {a_ok / max(a_n, 1):.4f} ({a_n}개) | 추론 대상 중 01 이 맞은 비율 {d_ok / max(d_n, 1):.3f} ({d_n}개)")


def log_row(out_dir, row):
    path = os.path.join(out_dir, "anchor_runs.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def floats(s):
    return [float(v) for v in str(s).split(",") if str(v).strip()]


def tag(args):
    """후보 캐시 이름 (고정 기준·참조 모델·후보 수·빔·고정 풀기 설정이 같으면 재사용)."""
    t = f"{args.anchor}{'' if args.anchor == 'agree' else args.min_len}_{'-'.join(args.refs)}_k{args.nbest}"
    return t + (f"_b{args.cand_beam}" if args.cand_beam else "") + ("_uo" if args.unanchor_oov else "")


def out_tag(args):
    """결과 파일 이름. --name 을 주면 꼬리표가 붙어 다른 실험 결과를 덮어쓰지 않는다."""
    return tag(args) + (f"_{args.name}" if args.name else "")


def load_preds(path, ids, col):
    df = pd.read_csv(path).set_index("ID")
    return [str(v) if pd.notna(v) else "" for v in df.loc[ids, col]]


def run(args):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    o = args.out_dir
    if args.mode == "val":
        tr, df = split_train(full, args.val_ratio, args.seed)
        if args.val_limit:
            df = df.iloc[: args.val_limit]
        ids = list(df.ID)
        p01s = load_preds(os.path.join(o, "val_pred.csv"), ids, "pred")
        refs = [load_preds(os.path.join(o, REFS[r][0]), ids, "pred") for r in args.refs]
    else:
        tr, df = full, pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        ids = list(df.ID)
        p01s = load_preds(os.path.join(o, "submission_noisy_channel.csv"), ids, "output")
        refs = []
        for r in args.refs:
            path = os.path.join(o, REFS[r][1])
            if not os.path.exists(path):
                raise SystemExit(f"{path} 가 없습니다. 먼저 실행: {REFS[r][2]}")
            refs.append(load_preds(path, ids, "output"))
    xs = list(df.input)
    sfx = f"_n{args.val_limit}" if args.mode == "val" and args.val_limit else ""
    ipath = os.path.join(o, f"anchor_items_{args.mode}{sfx}_{tag(args)}.pkl")
    if os.path.exists(ipath):
        with open(ipath, "rb") as f:
            items = pickle.load(f)
        print(f"  캐시 사용: {ipath}")
    else:
        items = make_items(tr, xs, p01s, refs, args)
        with open(ipath, "wb") as f:
            pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)

    # train 정답에 나온 어절 빈도 (검증 때는 학습 분할만, test 때는 train 전체)
    freq = collections.Counter(m.group() for y in tr.output for m in re.finditer(r"[가-힣]+", str(y)))
    lm = LMScorer(args.lm, args.batch_tokens)
    if args.mode == "val":
        gts = list(df.output)
        anchor_report(xs, p01s, items, gts)
        base = nc.mean_f1(p01s, gts)
        orc = nc.mean_f1(oracle(p01s, items, gts), gts)
        print(f"  01 F1 {base:.4f} | 추론 대상에서 후보 중 최선을 고를 때 상한 {orc:.4f}")
        best = (-1, None, None)
        grid = [(w, wf, mg) for w in floats(args.weights) for wf in floats(args.freq_weights)
                for mg in floats(args.margins)]
        for w, wf, mg in grid:
            t = time.time()
            preds = resolve(xs, p01s, items, lm, w, args, wf, mg, freq)
            f1 = nc.mean_f1(preds, gts)
            log_row(o, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_val": len(df), "anchor": args.anchor,
                        "min_len": args.min_len, "refs": "-".join(args.refs), "nbest": args.nbest, "order": args.order,
                        "passes": args.passes, "ctx": args.ctx, "lm": args.lm, "w": w, "f1": round(f1, 5),
                        "base_f1": round(base, 5), "oracle_f1": round(orc, 5), "sec": round(time.time() - t),
                        "w_freq": wf, "margin": mg, "name": args.name})
            mark = ""
            if f1 > best[0]:
                best, mark = (f1, (w, wf, mg), preds), " *"
            print(f"  w {w:<5} w_freq {wf:<5} margin {mg:<4} F1 {f1:.4f}{mark}  "
                  f"({time.time() - t:.0f}s, LM 누적 {lm.n_forward}회)", flush=True)
        f1, (w, wf, mg), preds = best
        print(f"best F1 {f1:.4f} (w {w}, w_freq {wf}, margin {mg}) | 01 대비 {f1 - base:+.4f} "
              f"| 07 재순위화 0.9843, 08 기본 0.9894 와 비교")
        df.assign(pred=preds, f1=[nc.f1_score(p, g) for p, g in zip(preds, gts)]).to_csv(
            os.path.join(o, f"val_pred_anchor_{out_tag(args)}.csv"), index=False, encoding="utf-8-sig")
        if not args.val_limit:
            with open(os.path.join(o, f"best_anchor_{out_tag(args)}.json"), "w", encoding="utf-8") as fp:
                json.dump({**{k: getattr(args, k) for k in ("anchor", "min_len", "refs", "nbest", "order", "passes",
                                                             "ctx", "lm")},
                           "w": w, "w_freq": wf, "margin": mg, "f1": f1, "base_f1": base, "oracle_f1": orc,
                           "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, ensure_ascii=False, indent=1)
    else:
        bpath = os.path.join(o, f"best_anchor_{out_tag(args)}.json")
        if not os.path.exists(bpath):
            raise SystemExit(f"{bpath} 가 없습니다. 먼저 같은 옵션으로 --mode val (val_limit 없이) 실행.")
        with open(bpath, encoding="utf-8") as fp:
            best = json.load(fp)
        wf, mg = best.get("w_freq", 0.0), best.get("margin", 0.0)
        preds = resolve(xs, p01s, items, lm, best["w"], args, wf, mg, freq)
        print(f"  w {best['w']}, w_freq {wf}, margin {mg} (val F1 {best['f1']:.4f}) | 01 과 다른 문장 "
              f"{sum(a != b for a, b in zip(preds, p01s))}/{len(preds)}")
        sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
        sub["output"] = sub.ID.map(dict(zip(ids, preds)))
        out = os.path.join(o, f"submission_anchor_{out_tag(args)}.csv")
        sub.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "test"], default="val")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--refs", default="v2", help=f"참조 모델 (쉼표로 여러 개): {list(REFS)}")
    ap.add_argument("--anchor", choices=["agree", "length", "both"], default="agree",
                    help="agree: 01·참조 일치 어절 고정 / length: min_len 이상 고정(사용자 원안) / both: 둘 중 하나")
    ap.add_argument("--min_len", type=int, default=4)
    ap.add_argument("--nbest", type=int, default=6, help="불일치 어절마다 01 이 추가로 내는 후보 수")
    ap.add_argument("--order", choices=["length", "position"], default="length")
    ap.add_argument("--passes", type=int, default=2, help="1 = 정방향만, 2 = 정방향 후 역방향 재결정")
    ap.add_argument("--ctx", type=int, default=120, help="LM 에 넣을 앞뒤 문맥 글자 수 (각각)")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--batch_tokens", type=int, default=4096)
    ap.add_argument("--weights", default="1.5", help="문맥(LM) 가중치 탐색 범위 (08 탐색: 0.75~5 가 거의 평평, 최고 1.5)")
    ap.add_argument("--freq_weights", default="0", help="09: train 어절 빈도 log(1+횟수) 가중치 탐색 범위 (0 = 끔)")
    ap.add_argument("--margins", default="0", help="09: 01 의 답을 바꾸는 데 필요한 최소 점수 차 탐색 범위 (0 = 끔)")
    ap.add_argument("--name", default="", help="결과 파일 꼬리표 (예: 09). 후보 캐시는 공유")
    ap.add_argument("--aug_file", default="",
                    help="10단계: 01 후보 생성에 쓸 생성 문장 파일 (지정하면 best_params.json 의 aug_file 대신 사용)")
    ap.add_argument("--unanchor_oov", action="store_true",
                    help="09-A: 01 의 답이 train 정답에 없는 어절은 일치해도 고정하지 않고 추론 대상에 넣음")
    ap.add_argument("--cand_beam", type=int, default=0,
                    help="09-B: 후보 생성용 01 빔 크기 (0 = 01 설정 그대로). --nbest 와 함께 키움. 예: 64")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    args.refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    print(vars(args))
    run(args)


if __name__ == "__main__":
    main()
