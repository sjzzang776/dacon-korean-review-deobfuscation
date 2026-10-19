"""
난독화 한글 리뷰 복원 - 12단계: LLM 없는 '고정 + 문맥 추론'

08~10 (anchor_infill.py) 과 같은 구조에서 문맥 판단만 바꾼다.
  고정·후보 : anchor_infill.py 그대로 (01 과 v2 태거가 일치하는 어절 고정, 01 의 상위 후보). 후보 캐시도 공유.
  문맥      : 공개 LLM 대신 infill_lm.py 의 빈칸 채우기 모델 (train 정답만으로 처음부터 학습)
  점수      = 채널 유사도 log P(입력 어절 | 후보) + w × log P(후보 | 앞뒤 문맥) [+ w_freq × log(1 + train 빈도)]

사전학습 모델·LLM·LLM 생성 문장을 전혀 쓰지 않는다 (--out_dir outputs = 증강 없는 01 기준).

사용 예 (Colab, GPU):
    !python -u anchor_infill12.py --mode val  --unanchor_oov --cand_beam 64 --nbest 20
    !python -u anchor_infill12.py --mode test --unanchor_oov --cand_beam 64 --nbest 20
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
from infill_lm import InfillScorer
from neural_tagger import split_train


def resolve(p01s, items, scorer, w, args, wf=0.0, margin=0.0, freq=None):
    """anchor_infill.resolve 와 같은 순서·규칙. 문맥 점수만 빈칸 채우기 모델로 계산."""
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
            groups, owners = [], []
            for si, seq in enumerate(seqs):
                if step >= len(seq):
                    continue
                it = items[si]["items"][seq[step]]
                s, e = it["s"], it["e"]
                lo, hi = max(0, s - args.ctx), min(len(cur[si]), e + args.ctx)
                groups.append(("".join(cur[si][lo:s]), it["cands"], "".join(cur[si][e:hi])))
                owners.append((si, seq[step]))
            if not groups:
                continue
            for (si, ii), lps in zip(owners, scorer.score_spans(groups)):
                it = items[si]["items"][ii]
                sc = [sim + w * lp + (wf * math.log1p(freq.get(c, 0)) if wf else 0.0)
                      for sim, lp, c in zip(it["sim"], lps, it["cands"])]
                k = max(range(len(sc)), key=sc.__getitem__)
                if margin and k != 0 and sc[k] - sc[0] < margin:  # 후보 0 = 01 의 답
                    k = 0
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
    ckpt = args.ckpt or os.path.join(o, "infill_best.pt" if args.mode == "val" else "infill_full.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"{ckpt} 가 없습니다. 먼저 실행: python -u infill_lm.py --mode train"
                         + (" --full --epochs <검증에서 가장 좋았던 epoch>" if args.mode == "test" else ""))
    scorer = InfillScorer(ckpt, args.batch_tokens)
    out_tag = ai.tag(args) + f"_{args.name}"
    if args.mode == "val":
        gts = list(df.output)
        ai.anchor_report(xs, p01s, items, gts)
        base = nc.mean_f1(p01s, gts)
        orc = nc.mean_f1(ai.oracle(p01s, items, gts), gts)
        print(f"  01 F1 {base:.4f} | 추론 대상에서 후보 중 최선을 고를 때 상한 {orc:.4f}")
        best = (-1, None, None)
        grid = [(w, wf, mg) for w in ai.floats(args.weights) for wf in ai.floats(args.freq_weights)
                for mg in ai.floats(args.margins)]
        for w, wf, mg in grid:
            t = time.time()
            preds = resolve(p01s, items, scorer, w, args, wf, mg, freq)
            f1 = nc.mean_f1(preds, gts)
            ai.log_row(o, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_val": len(df), "anchor": args.anchor,
                           "min_len": args.min_len, "refs": "-".join(args.refs), "nbest": args.nbest,
                           "order": args.order, "passes": args.passes, "ctx": args.ctx, "lm": ckpt, "w": w,
                           "f1": round(f1, 5), "base_f1": round(base, 5), "oracle_f1": round(orc, 5),
                           "sec": round(time.time() - t), "w_freq": wf, "margin": mg, "name": args.name})
            mark = ""
            if f1 > best[0]:
                best, mark = (f1, (w, wf, mg), preds), " *"
            print(f"  w {w:<5} w_freq {wf:<5} margin {mg:<4} F1 {f1:.4f}{mark}  "
                  f"({time.time() - t:.0f}s, 모델 누적 {scorer.n_forward}회)", flush=True)
        f1, (w, wf, mg), preds = best
        print(f"best F1 {f1:.4f} (w {w}, w_freq {wf}, margin {mg}) | 01 대비 {f1 - base:+.4f} "
              f"| LLM 없는 기존 최고 0.9834 (01), LLM 사용 09 는 0.9908")
        df.assign(pred=preds, f1=[nc.f1_score(p, g) for p, g in zip(preds, gts)]).to_csv(
            os.path.join(o, f"val_pred_anchor_{out_tag}.csv"), index=False, encoding="utf-8-sig")
        if not args.val_limit:
            with open(os.path.join(o, f"best_anchor_{out_tag}.json"), "w", encoding="utf-8") as fp:
                json.dump({**{k: getattr(args, k) for k in ("anchor", "min_len", "refs", "nbest", "order", "passes",
                                                             "ctx")},
                           "scorer": ckpt, "w": w, "w_freq": wf, "margin": mg, "f1": f1, "base_f1": base,
                           "oracle_f1": orc, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, ensure_ascii=False,
                          indent=1)
    else:
        bpath = os.path.join(o, f"best_anchor_{out_tag}.json")
        if not os.path.exists(bpath):
            raise SystemExit(f"{bpath} 가 없습니다. 먼저 같은 옵션으로 --mode val (val_limit 없이) 실행.")
        with open(bpath, encoding="utf-8") as fp:
            best = json.load(fp)
        wf, mg = best.get("w_freq", 0.0), best.get("margin", 0.0)
        preds = resolve(p01s, items, scorer, best["w"], args, wf, mg, freq)
        print(f"  w {best['w']}, w_freq {wf}, margin {mg} (val F1 {best['f1']:.4f}) | 01 과 다른 문장 "
              f"{sum(a != b for a, b in zip(preds, p01s))}/{len(preds)}")
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
    ap.add_argument("--ctx", type=int, default=100, help="앞뒤 문맥 글자 수 (각각). 학습 때 최대 길이 256 안에 들어오게")
    ap.add_argument("--ckpt", default="", help="빈칸 채우기 모델 (기본: val 은 infill_best.pt, test 는 infill_full.pt)")
    ap.add_argument("--batch_tokens", type=int, default=65536)
    ap.add_argument("--weights", default="0.5,0.75,1.0,1.5,2.0,3.0", help="문맥 가중치 탐색 범위")
    ap.add_argument("--freq_weights", default="0", help="train 어절 빈도 log(1+횟수) 가중치 탐색 범위 (0 = 끔)")
    ap.add_argument("--margins", default="0", help="01 의 답을 바꾸는 데 필요한 최소 점수 차 탐색 범위 (0 = 끔)")
    ap.add_argument("--name", default="12", help="결과 파일 꼬리표")
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
