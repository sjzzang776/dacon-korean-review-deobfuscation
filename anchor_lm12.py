"""
난독화 한글 리뷰 복원 - 12b: LLM 없는 '고정 + 문맥 추론' (글자 언어모델 + 문자 5-gram)

고정·후보는 anchor_infill.py 그대로 (09 의 후보 캐시 재사용). 문맥 점수만 LLM 없이 계산한다.

  점수 = 채널 유사도 log P(입력 어절 | 후보)
       + w    × 글자 언어모델(char_lm.py, 정방향·역방향 평균)의 앞뒤 문맥 창 log P
       + w_kn × 01 의 문자 5-gram 이 본 후보 주변 log P
       + w_freq × log(1 + train 에서 그 어절이 나온 횟수)
  margin : 01 의 답(후보 0)을 바꾸려면 점수가 이만큼 높아야 한다. 크게 주면 01 과 같아지므로 하한이 01 이다.

탐색: 1단계 (w, w_kn, w_freq) 조합을 margin 0 으로, 2단계 그 최고 조합에서 margin 을 탐색.

사용 예 (Colab, GPU):
    !python -u anchor_lm12.py --mode val  --unanchor_oov --cand_beam 64 --nbest 20
    !python -u anchor_lm12.py --mode test --unanchor_oov --cand_beam 64 --nbest 20
"""
import argparse
import collections
import json
import math
import os
import pickle
import re
import time

import pandas as pd

import anchor_infill as ai
import noisy_channel as nc
from char_lm import CharLMScorer
from neural_tagger import split_train


class KNLocal:
    """문자 5-gram 으로 후보와 그 뒤 n-1 글자만 채점 (나머지 글자는 후보끼리 같으므로 생략)."""

    def __init__(self, texts, order=5):
        self.lm, self.n, self.cache = nc.KNLM(texts, order=order), order, {}

    def __call__(self, left, cand, right, at_start, at_end):
        k = self.n - 1
        h = ((nc.BOS * k if at_start else "") + left)[-k:]
        seq = cand + right[:k] + (nc.EOS if at_end and len(right) < k else "")
        key = (h, seq)
        v = self.cache.get(key)
        if v is None:
            v, s = 0.0, h + seq
            for i in range(len(h), len(s)):
                v += self.lm.logp(s[max(0, i - k) : i], s[i])
            self.cache[key] = v
        return v


def resolve(p01s, items, nn, kn, w, wk, wf, margin, freq, args):
    """anchor_infill.resolve 와 같은 순서·규칙. 문맥 점수만 글자 언어모델 + 5-gram 으로 계산."""
    cur = [list(p) for p in p01s]
    orders = []
    for res in items:
        its = list(range(len(res["items"])))
        if args.order == "length":
            its.sort(key=lambda k: (-(res["items"][k]["e"] - res["items"][k]["s"]), res["items"][k]["s"]))
        orders.append(its)
    for ps in range(args.passes):
        seqs = [o if ps % 2 == 0 else list(reversed(o)) for o in orders]
        for step in range(max((len(o) for o in seqs), default=0)):
            texts, knv, owners = [], [], []
            for si, seq in enumerate(seqs):
                if step >= len(seq):
                    continue
                it = items[si]["items"][seq[step]]
                s, e = it["s"], it["e"]
                lo, hi = max(0, s - args.ctx), min(len(cur[si]), e + args.ctx)
                left, right = "".join(cur[si][lo:s]), "".join(cur[si][e:hi])
                for k, c in enumerate(it["cands"]):
                    texts.append(left + c + right)
                    knv.append(kn(left, c, right, lo == 0, hi == len(cur[si])) if wk else 0.0)
                    owners.append((si, seq[step], k))
            if not texts:
                continue
            nnv = nn(texts) if w else [0.0] * len(texts)
            best, score01 = {}, {}
            for (si, ii, k), a, b in zip(owners, nnv, knv):
                it = items[si]["items"][ii]
                sc = it["sim"][k] + w * a + wk * b
                if wf:
                    sc += wf * math.log1p(freq.get(it["cands"][k], 0))
                if k == 0:
                    score01[(si, ii)] = sc
                if (si, ii) not in best or sc > best[(si, ii)][0]:
                    best[(si, ii)] = (sc, k)
            for (si, ii), (sc, k) in best.items():
                if margin and k != 0 and sc - score01[(si, ii)] < margin:
                    k = 0
                it = items[si]["items"][ii]
                cur[si][it["s"] : it["e"]] = list(it["cands"][k])
    return ["".join(c) for c in cur]


def run(args):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    o = args.out_dir
    if args.mode == "val":
        tr, df = split_train(full, args.val_ratio, args.seed)
        if args.val_limit:
            df = df.iloc[: args.val_limit]
        ids = list(df.ID)
        p01s = ai.load_preds(os.path.join(o, "val_pred.csv"), ids, "pred")
        refs = [ai.load_preds(os.path.join(o, ai.REFS[r][0]), ids, "pred") for r in args.refs]
    else:
        tr, df = full, pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        ids = list(df.ID)
        p01s = ai.load_preds(os.path.join(o, "submission_noisy_channel.csv"), ids, "output")
        refs = []
        for r in args.refs:
            path = os.path.join(o, ai.REFS[r][1])
            if not os.path.exists(path):
                raise SystemExit(f"{path} 가 없습니다. 먼저 실행: {ai.REFS[r][2]}")
            refs.append(ai.load_preds(path, ids, "output"))
    xs = list(df.input)
    sfx = f"_n{args.val_limit}" if args.mode == "val" and args.val_limit else ""
    ipath = os.path.join(o, f"anchor_items_{args.mode}{sfx}_{ai.tag(args)}.pkl")
    if os.path.exists(ipath):
        with open(ipath, "rb") as f:
            items = pickle.load(f)
        print(f"  캐시 사용: {ipath}")
    else:
        items = ai.make_items(tr, xs, p01s, refs, args)
        with open(ipath, "wb") as f:
            pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)

    freq = collections.Counter(m.group() for y in tr.output for m in re.finditer(r"[가-힣]+", str(y)))
    ckpt = args.ckpt or os.path.join(o, "charlm_best.pt" if args.mode == "val" else "charlm_full.pt")
    out_tag = ai.tag(args) + f"_{args.name}"
    bpath = os.path.join(o, f"best_anchor_{out_tag}.json")
    if args.mode == "val":
        need_nn = any(ai.floats(args.weights))
    else:
        if not os.path.exists(bpath):
            raise SystemExit(f"{bpath} 가 없습니다. 먼저 같은 옵션으로 --mode val (val_limit 없이) 실행.")
        with open(bpath, encoding="utf-8") as fp:
            b = json.load(fp)
        need_nn = b["w"] > 0  # 검증에서 글자 언어모델 가중치가 0 이 최고였으면 모델 없이 진행
    nn = None
    if need_nn:
        if not os.path.exists(ckpt):
            raise SystemExit(f"{ckpt} 가 없습니다. 먼저 실행: python -u char_lm.py"
                             + (" --full --epochs <검증에서 가장 좋았던 epoch>" if args.mode == "test" else ""))
        nn = CharLMScorer(ckpt, args.batch_tokens)
    kn = KNLocal([str(y) for y in tr.output], order=ai.nc_params(o)["order"])
    if args.mode == "val":
        gts = list(df.output)
        ai.anchor_report(xs, p01s, items, gts)
        base = nc.mean_f1(p01s, gts)
        orc = nc.mean_f1(ai.oracle(p01s, items, gts), gts)
        print(f"  01 F1 {base:.4f} | 추론 대상에서 후보 중 최선을 고를 때 상한 {orc:.4f}")
        best = (-1, None, None)

        def trial(w, wk, wf, mg):
            nonlocal best
            t = time.time()
            preds = resolve(p01s, items, nn, kn, w, wk, wf, mg, freq, args)
            f1 = nc.mean_f1(preds, gts)
            ai.log_row(o, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_val": len(df), "anchor": args.anchor,
                           "min_len": args.min_len, "refs": "-".join(args.refs), "nbest": args.nbest,
                           "order": args.order, "passes": args.passes, "ctx": args.ctx, "lm": ckpt, "w": w,
                           "f1": round(f1, 5), "base_f1": round(base, 5), "oracle_f1": round(orc, 5),
                           "sec": round(time.time() - t), "w_freq": wf, "margin": mg, "name": args.name, "w_kn": wk})
            mark = ""
            if f1 > best[0]:
                best, mark = (f1, (w, wk, wf, mg), preds), " *"
            print(f"  w {w:<5} w_kn {wk:<5} w_freq {wf:<5} margin {mg:<4} F1 {f1:.4f}{mark}  "
                  f"({time.time() - t:.0f}s, 모델 누적 {nn.n_forward if nn else 0}회)", flush=True)

        print("[1단계] 가중치 조합 (margin 0)")
        for w in ai.floats(args.weights):
            for wk in ai.floats(args.kn_weights):
                for wf in ai.floats(args.freq_weights):
                    trial(w, wk, wf, 0.0)
        w, wk, wf, _ = best[1]
        print(f"[2단계] 최고 조합 (w {w}, w_kn {wk}, w_freq {wf}) 에서 01 유지 문턱")
        for mg in ai.floats(args.margins):
            if mg:
                trial(w, wk, wf, mg)
        f1, (w, wk, wf, mg), preds = best
        print(f"best F1 {f1:.4f} (w {w}, w_kn {wk}, w_freq {wf}, margin {mg}) | 01 대비 {f1 - base:+.4f} "
              f"| LLM 없는 기존 최고 0.9834 (01), 12 (빈칸 채우기) 0.9583, LLM 사용 09 는 0.9908")
        df.assign(pred=preds, f1=[nc.f1_score(p, g) for p, g in zip(preds, gts)]).to_csv(
            os.path.join(o, f"val_pred_anchor_{out_tag}.csv"), index=False, encoding="utf-8-sig")
        if not args.val_limit:
            with open(bpath, "w", encoding="utf-8") as fp:
                json.dump({**{k: getattr(args, k) for k in ("anchor", "min_len", "refs", "nbest", "order", "passes",
                                                             "ctx")},
                           "scorer": ckpt, "w": w, "w_kn": wk, "w_freq": wf, "margin": mg, "f1": f1, "base_f1": base,
                           "oracle_f1": orc, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, ensure_ascii=False,
                          indent=1)
    else:
        preds = resolve(p01s, items, nn, kn, b["w"], b["w_kn"], b["w_freq"], b["margin"], freq, args)
        print(f"  w {b['w']}, w_kn {b['w_kn']}, w_freq {b['w_freq']}, margin {b['margin']} (val F1 {b['f1']:.4f}) "
              f"| 01 과 다른 문장 {sum(x != y for x, y in zip(preds, p01s))}/{len(preds)}")
        sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
        sub["output"] = sub.ID.map(dict(zip(ids, preds)))
        out = os.path.join(o, f"submission_anchor_{out_tag}.csv")
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
    ap.add_argument("--refs", default="v2")
    ap.add_argument("--anchor", choices=["agree", "length", "both"], default="agree")
    ap.add_argument("--min_len", type=int, default=4)
    ap.add_argument("--nbest", type=int, default=20)
    ap.add_argument("--order", choices=["length", "position"], default="length")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=80, help="앞뒤 문맥 글자 수 (각각)")
    ap.add_argument("--ckpt", default="", help="글자 언어모델 (기본: val 은 charlm_best.pt, test 는 charlm_full.pt)")
    ap.add_argument("--batch_tokens", type=int, default=131072)
    ap.add_argument("--weights", default="0,0.5,1.0,1.5", help="글자 언어모델 가중치 탐색 범위")
    ap.add_argument("--kn_weights", default="0,0.5,1.0", help="문자 5-gram 가중치 탐색 범위")
    ap.add_argument("--freq_weights", default="0,1.0", help="train 어절 빈도 log(1+횟수) 가중치 탐색 범위")
    ap.add_argument("--margins", default="1,2,4", help="2단계: 01 의 답을 바꾸는 데 필요한 최소 점수 차 탐색 범위")
    ap.add_argument("--name", default="12b", help="결과 파일 꼬리표")
    ap.add_argument("--aug_file", default="")
    ap.add_argument("--unanchor_oov", action="store_true")
    ap.add_argument("--cand_beam", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    args.refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    print(vars(args))
    run(args)


if __name__ == "__main__":
    main()
