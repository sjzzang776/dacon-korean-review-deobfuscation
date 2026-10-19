"""
난독화 한글 리뷰 복원 - 14단계: 11b (LLM) 와 13b (KoCharELECTRA) 합치기

검증에서 13b (0.9922) 와 11b (0.9915) 는 서로 다른 문장을 맞혔다 (13b 가 나은 문장 153개, 11b 가 나은 문장 136개).
11b 의 재료 위에 13b 의 점수 항을 얹는다.

  재료 (11b, outputs10/) : 생성 문장으로 넓힌 01, 참조 v2·v3 태거, 그 후보 캐시
  점수 = 채널 유사도
       + w_llm × 미세조정 Qwen 의 앞뒤 문맥 창 log P (창 ±ctx 글자)                    ← 11b
       + w × 글자 언어모델 + w_kn × 문자 5-gram + w_rtd × KoCharELECTRA discriminator   ← 13b (창 ±ctx_small 글자)

탐색: w_llm × (글자 언어모델·5-gram 켜기/끄기) × w_rtd.
      (w_llm 1.0, 나머지 0) 은 11b 재현 (0.9915) 이어야 한다.

사용 예 (Colab, GPU):
    !python -u anchor_combo14.py --mode val
    !python -u anchor_combo14.py --mode test --aug_file outputs/gen_reviews_full.txt
"""
import argparse
import json
import os
import pickle
import time

import pandas as pd

import anchor_infill as ai
import noisy_channel as nc
from anchor_lm12 import KNLocal
from neural_tagger import split_train

KEYS = ("llm", "nn", "kn", "rtd")


def resolve(p01s, items, sc, wt, args):
    """sc: {'llm','nn','kn','rtd'} 채점기 (없으면 None), wt: 같은 키의 가중치. 순서·규칙은 anchor_infill.resolve 와 같다."""
    cur = [list(p) for p in p01s]
    orders = []
    for res in items:
        its = list(range(len(res["items"])))
        if args.order == "length":
            its.sort(key=lambda k: (-(res["items"][k]["e"] - res["items"][k]["s"]), res["items"][k]["s"]))
        orders.append(its)
    small = wt["nn"] or wt["kn"] or wt["rtd"]
    for ps in range(args.passes):
        seqs = [o if ps % 2 == 0 else list(reversed(o)) for o in orders]
        for step in range(max((len(o) for o in seqs), default=0)):
            big, texts, reqs, knv, owners = [], [], [], [], []
            for si, seq in enumerate(seqs):
                if step >= len(seq):
                    continue
                it = items[si]["items"][seq[step]]
                s, e, n = it["s"], it["e"], len(cur[si])
                if wt["llm"]:
                    lo, hi = max(0, s - args.ctx), min(n, e + args.ctx)
                    bl, br = "".join(cur[si][lo:s]), "".join(cur[si][e:hi])
                if small:
                    lo, hi = max(0, s - args.ctx_small), min(n, e + args.ctx_small)
                    left, right = "".join(cur[si][lo:s]), "".join(cur[si][e:hi])
                for k, c in enumerate(it["cands"]):
                    if wt["llm"]:
                        big.append(bl + c + br)
                    if small:
                        texts.append(left + c + right)
                        reqs.append((left + c + right, len(left), len(left) + len(c)))
                        knv.append(sc["kn"](left, c, right, lo == 0, hi == n) if wt["kn"] else 0.0)
                    owners.append((si, seq[step], k))
            if not owners:
                continue
            zero = [0.0] * len(owners)
            llv = sc["llm"](big) if wt["llm"] else zero
            nnv = sc["nn"](texts) if wt["nn"] else zero
            rtv = sc["rtd"](reqs) if wt["rtd"] else zero
            knv = knv if wt["kn"] else zero
            best = {}
            for (si, ii, k), a, b, c, d in zip(owners, llv, nnv, knv, rtv):
                v = items[si]["items"][ii]["sim"][k] + wt["llm"] * a + wt["nn"] * b + wt["kn"] * c + wt["rtd"] * d
                if (si, ii) not in best or v > best[(si, ii)][0]:
                    best[(si, ii)] = (v, k)
            for (si, ii), (_, k) in best.items():
                it = items[si]["items"][ii]
                cur[si][it["s"] : it["e"]] = list(it["cands"][k])
    return ["".join(c) for c in cur]


def run(args):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    o, aux = args.out_dir, args.aux_dir
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
        refs = [ai.load_preds(os.path.join(o, ai.REFS[r][1]), ids, "output") for r in args.refs]
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

    out_tag = ai.tag(args) + f"_{args.name}"
    bpath = os.path.join(o, f"best_anchor_{out_tag}.json")
    if args.mode == "val":
        llm_ws, rtd_ws = ai.floats(args.llm_weights), ai.floats(args.rtd_weights)
        base = [tuple(float(v) for v in p.split(":")) for p in args.base_weights.split(",")]  # (w, w_kn) 쌍
        need = {"llm": any(llm_ws), "nn": any(w for w, _ in base), "kn": any(k for _, k in base), "rtd": any(rtd_ws)}
    else:
        if not os.path.exists(bpath):
            raise SystemExit(f"{bpath} 가 없습니다. 먼저 --mode val (val_limit 없이) 실행.")
        with open(bpath, encoding="utf-8") as fp:
            b = json.load(fp)
        need = {"llm": b["w_llm"] > 0, "nn": b["w"] > 0, "kn": b["w_kn"] > 0, "rtd": b["w_rtd"] > 0}

    split = "split" if args.mode == "val" else "full"
    lm_name = args.lm or f"models/qwen_ft_{split}"
    sc = dict.fromkeys(KEYS)
    if need["llm"]:
        if not os.path.exists(lm_name):
            raise SystemExit(f"{lm_name} 가 없습니다. 먼저 실행: python -u lm_finetune.py"
                             + (" --full" if args.mode == "test" else "") + " --merge_only")
        sc["llm"] = ai.LMScorer(lm_name, args.batch_tokens)
    if need["nn"]:
        from char_lm import CharLMScorer
        ckpt = os.path.join(aux, "charlm_best.pt" if args.mode == "val" else "charlm_full.pt")
        if not os.path.exists(ckpt):
            raise SystemExit(f"{ckpt} 가 없습니다 (12 노트북의 글자 언어모델).")
        sc["nn"] = CharLMScorer(ckpt)
    if need["kn"]:
        sc["kn"] = KNLocal([str(y) for y in tr.output], order=ai.nc_params(o)["order"])
    if need["rtd"]:
        import electra13 as el
        sc["rtd"] = el.RTDScorer(el.DISC, args.batch_tokens_small, args.rtd_pad)
    nfw = lambda: " ".join(f"{k} {sc[k].n_forward}" for k in ("llm", "nn", "rtd") if sc[k] is not None)

    if args.mode == "val":
        gts = list(df.output)
        ai.anchor_report(xs, p01s, items, gts)
        f01 = nc.mean_f1(p01s, gts)
        orc = nc.mean_f1(ai.oracle(p01s, items, gts), gts)
        print(f"  01 F1 {f01:.4f} | 추론 대상에서 후보 중 최선을 고를 때 상한 {orc:.4f}")
        best = (-1, None, None)
        for wl in llm_ws:
            for w, wk in base:
                for wr in rtd_ws:
                    t = time.time()
                    preds = resolve(p01s, items, sc, {"llm": wl, "nn": w, "kn": wk, "rtd": wr}, args)
                    f1 = nc.mean_f1(preds, gts)
                    ai.log_row(o, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_val": len(df),
                                   "anchor": args.anchor, "min_len": args.min_len, "refs": "-".join(args.refs),
                                   "nbest": args.nbest, "order": args.order, "passes": args.passes, "ctx": args.ctx,
                                   "lm": lm_name, "w": wl, "f1": round(f1, 5), "base_f1": round(f01, 5),
                                   "oracle_f1": round(orc, 5), "sec": round(time.time() - t), "w_freq": 0.0,
                                   "margin": 0.0, "name": args.name, "w_nn": w, "w_kn": wk, "w_rtd": wr,
                                   "ctx_small": args.ctx_small, "rtd_pad": args.rtd_pad})
                    mark = ""
                    if f1 > best[0]:
                        best, mark = (f1, (wl, w, wk, wr), preds), " *"
                    print(f"  w_llm {wl:<4} w {w:<4} w_kn {wk:<4} w_rtd {wr:<5} F1 {f1:.4f}{mark}  "
                          f"({time.time() - t:.0f}s, 누적 {nfw()})", flush=True)
        f1, (wl, w, wk, wr), preds = best
        print(f"best F1 {f1:.4f} (w_llm {wl}, w {w}, w_kn {wk}, w_rtd {wr}) | 01 대비 {f1 - f01:+.4f} "
              f"| 11b (Qwen) 0.9915, 13b (KoCharELECTRA) 0.9922")
        df.assign(pred=preds, f1=[nc.f1_score(p, g) for p, g in zip(preds, gts)]).to_csv(
            os.path.join(o, f"val_pred_anchor_{out_tag}.csv"), index=False, encoding="utf-8-sig")
        if not args.val_limit:
            with open(bpath, "w", encoding="utf-8") as fp:
                json.dump({**{k: getattr(args, k) for k in ("anchor", "min_len", "refs", "nbest", "order", "passes",
                                                             "ctx", "ctx_small", "rtd_pad")},
                           "lm": lm_name, "w_llm": wl, "w": w, "w_kn": wk, "w_rtd": wr, "f1": f1, "base_f1": f01,
                           "oracle_f1": orc, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, ensure_ascii=False,
                          indent=1)
    else:
        for k in ("ctx", "ctx_small", "rtd_pad"):
            setattr(args, k, b.get(k, getattr(args, k)))
        if sc["rtd"] is not None:
            sc["rtd"].pad = args.rtd_pad
        preds = resolve(p01s, items, sc, {"llm": b["w_llm"], "nn": b["w"], "kn": b["w_kn"], "rtd": b["w_rtd"]}, args)
        print(f"  w_llm {b['w_llm']}, w {b['w']}, w_kn {b['w_kn']}, w_rtd {b['w_rtd']} (val F1 {b['f1']:.4f}) "
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
    ap.add_argument("--out_dir", default="outputs10", help="11b 의 재료(증강한 01, 태거 예측, 후보 캐시)와 결과 저장 위치")
    ap.add_argument("--aux_dir", default="outputs", help="글자 언어모델(charlm_*.pt) 위치")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--refs", default="v2,v3")
    ap.add_argument("--anchor", choices=["agree", "length", "both"], default="agree")
    ap.add_argument("--min_len", type=int, default=4)
    ap.add_argument("--nbest", type=int, default=20)
    ap.add_argument("--cand_beam", type=int, default=64)
    ap.add_argument("--no_unanchor_oov", dest="unanchor_oov", action="store_false")
    ap.add_argument("--order", choices=["length", "position"], default="length")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=120, help="LLM 문맥 창 (앞뒤 글자 수, 11b 와 동일)")
    ap.add_argument("--ctx_small", type=int, default=80, help="글자 언어모델·discriminator 문맥 창 (13b 와 동일)")
    ap.add_argument("--rtd_pad", type=int, default=3)
    ap.add_argument("--lm", default="", help="병합한 Qwen 경로 (기본: models/qwen_ft_split|full)")
    ap.add_argument("--llm_weights", default="0,0.5,1.0")
    ap.add_argument("--base_weights", default="0:0,0.5:0.5", help="(글자 언어모델 w):(5-gram w_kn) 쌍들, 쉼표로 구분")
    ap.add_argument("--rtd_weights", default="0,0.5,1.0")
    ap.add_argument("--batch_tokens", type=int, default=8192, help="LLM 배치 (토큰)")
    ap.add_argument("--batch_tokens_small", type=int, default=32768, help="discriminator 배치 (토큰)")
    ap.add_argument("--name", default="14")
    ap.add_argument("--aug_file", default="")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    args.refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    print(vars(args))
    run(args)


if __name__ == "__main__":
    main()
