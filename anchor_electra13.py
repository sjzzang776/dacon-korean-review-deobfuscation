"""
난독화 한글 리뷰 복원 - 13단계: 12b + 글자 단위 사전학습 인코더 (KoCharELECTRA)

고정·후보는 그대로 (09 의 후보 캐시), 12b 의 점수에 사전학습 모델 두 항을 더한다.

  점수 = 채널 유사도
       + w     × 글자 언어모델 (12b, train 만으로 학습)        ┐ 12b 의 최고 가중치를
       + w_kn  × 문자 5-gram (01)                              ┘ 그대로 가져와 고정
       + w_mlm × KoCharELECTRA generator 의 후보 글자 맞히기 log P (글자를 하나씩 가림)
       + w_rtd × KoCharELECTRA discriminator 의 '원래 글자' 점수 (후보 구간 ± rtd_pad 글자)

탐색: 1단계 (w_mlm, w_rtd) 조합. (0, 0) 은 12b 재현 (0.9871).
      2단계 최고 조합에서 12b 항(글자 언어모델, 5-gram)을 하나씩 빼 본다 → 사전학습 모델만으로 충분한지 확인.

사용 예 (Colab, GPU):
    !python -u anchor_electra13.py --mode val  --unanchor_oov --cand_beam 64 --nbest 20
    !python -u anchor_electra13.py --mode test --unanchor_oov --cand_beam 64 --nbest 20
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


def resolve(p01s, items, sc, wt, args):
    """sc: {'nn','kn','mlm','rtd'} 채점기 (없으면 None), wt: 같은 키의 가중치."""
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
            texts, reqs, knv, owners = [], [], [], []
            for si, seq in enumerate(seqs):
                if step >= len(seq):
                    continue
                it = items[si]["items"][seq[step]]
                s, e = it["s"], it["e"]
                lo, hi = max(0, s - args.ctx), min(len(cur[si]), e + args.ctx)
                left, right = "".join(cur[si][lo:s]), "".join(cur[si][e:hi])
                for k, c in enumerate(it["cands"]):
                    texts.append(left + c + right)
                    reqs.append((left + c + right, len(left), len(left) + len(c)))
                    knv.append(sc["kn"](left, c, right, lo == 0, hi == len(cur[si])) if wt["kn"] else 0.0)
                    owners.append((si, seq[step], k))
            if not texts:
                continue
            zero = [0.0] * len(texts)
            nnv = sc["nn"](texts) if wt["nn"] else zero
            mlv = sc["mlm"](reqs) if wt["mlm"] else zero
            rtv = sc["rtd"](reqs) if wt["rtd"] else zero
            best = {}
            for (si, ii, k), a, b, c, d in zip(owners, nnv, knv, mlv, rtv):
                v = items[si]["items"][ii]["sim"][k] + wt["nn"] * a + wt["kn"] * b + wt["mlm"] * c + wt["rtd"] * d
                if (si, ii) not in best or v > best[(si, ii)][0]:
                    best[(si, ii)] = (v, k)
            for (si, ii), (_, k) in best.items():
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
    else:
        tr, df = full, pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        ids = list(df.ID)
        p01s = ai.load_preds(os.path.join(o, "submission_noisy_channel.csv"), ids, "output")
    xs = list(df.input)
    sfx = f"_n{args.val_limit}" if args.mode == "val" and args.val_limit else ""
    ipath = os.path.join(o, f"anchor_items_{args.mode}{sfx}_{ai.tag(args)}.pkl")
    if not os.path.exists(ipath):
        raise SystemExit(f"{ipath} 가 없습니다 (09 노트북이 만든 후보 캐시).")
    with open(ipath, "rb") as f:
        items = pickle.load(f)
    print(f"  캐시 사용: {ipath}")

    split = "split" if args.mode == "val" else "full"
    out_tag = ai.tag(args) + f"_{args.name}"
    bpath = os.path.join(o, f"best_anchor_{out_tag}.json")
    if args.mode == "val":
        with open(os.path.join(o, f"best_anchor_{ai.tag(args)}_{args.base}.json"), encoding="utf-8") as fp:
            b = json.load(fp)
        w0, wk0 = b["w"], b["w_kn"]
        print(f"  12b 기준 ({args.base}): w {w0}, w_kn {wk0}, val F1 {b['f1']:.4f}")
        mlm_ws, rtd_ws = ai.floats(args.mlm_weights), ai.floats(args.rtd_weights)
        need = {"nn": w0 > 0, "kn": wk0 > 0, "mlm": any(mlm_ws), "rtd": any(rtd_ws)}
    else:
        if not os.path.exists(bpath):
            raise SystemExit(f"{bpath} 가 없습니다. 먼저 같은 옵션으로 --mode val (val_limit 없이) 실행.")
        with open(bpath, encoding="utf-8") as fp:
            b = json.load(fp)
        need = {"nn": b["w"] > 0, "kn": b["w_kn"] > 0, "mlm": b["w_mlm"] > 0, "rtd": b["w_rtd"] > 0}

    sc = {"nn": None, "kn": None, "mlm": None, "rtd": None}
    if need["nn"]:
        from char_lm import CharLMScorer
        ckpt = os.path.join(o, "charlm_best.pt" if args.mode == "val" else "charlm_full.pt")
        if not os.path.exists(ckpt):
            raise SystemExit(f"{ckpt} 가 없습니다 (12 노트북의 글자 언어모델).")
        sc["nn"] = CharLMScorer(ckpt)
    if need["kn"]:
        sc["kn"] = KNLocal([str(y) for y in tr.output], order=ai.nc_params(o)["order"])
    if need["mlm"] or need["rtd"]:
        import electra13 as el
    mlm_name = args.mlm or os.path.join(o, f"kochar_gen_{split}")
    if need["mlm"]:
        if not args.mlm and not os.path.exists(mlm_name):
            raise SystemExit(f"{mlm_name} 가 없습니다. 먼저 실행: python -u electra13.py --mode ft"
                             + (" --full --epochs <검증에서 가장 좋았던 epoch>" if args.mode == "test" else "")
                             + "  (미세조정 없이 쓰려면 --mlm monologg/kocharelectra-base-generator)")
        sc["mlm"] = el.MLMScorer(mlm_name, args.batch_tokens)
    if need["rtd"]:
        sc["rtd"] = el.RTDScorer(el.DISC, args.batch_tokens // 2, args.rtd_pad)
    nfw = lambda: " ".join(f"{k} {sc[k].n_forward}" for k in ("nn", "mlm", "rtd") if sc[k] is not None)

    if args.mode == "val":
        gts = list(df.output)
        ai.anchor_report(xs, p01s, items, gts)
        base = nc.mean_f1(p01s, gts)
        orc = nc.mean_f1(ai.oracle(p01s, items, gts), gts)
        print(f"  01 F1 {base:.4f} | 추론 대상에서 후보 중 최선을 고를 때 상한 {orc:.4f}")
        best = (-1, None, None)

        def trial(w, wk, wm, wr):
            nonlocal best
            t = time.time()
            preds = resolve(p01s, items, sc, {"nn": w, "kn": wk, "mlm": wm, "rtd": wr}, args)
            f1 = nc.mean_f1(preds, gts)
            ai.log_row(o, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_val": len(df), "anchor": args.anchor,
                           "min_len": args.min_len, "refs": "-".join(args.refs), "nbest": args.nbest,
                           "order": args.order, "passes": args.passes, "ctx": args.ctx, "lm": mlm_name, "w": w,
                           "f1": round(f1, 5), "base_f1": round(base, 5), "oracle_f1": round(orc, 5),
                           "sec": round(time.time() - t), "w_freq": 0.0, "margin": 0.0, "name": args.name,
                           "w_kn": wk, "w_mlm": wm, "w_rtd": wr, "rtd_pad": args.rtd_pad})
            mark = ""
            if f1 > best[0]:
                best, mark = (f1, (w, wk, wm, wr), preds), " *"
            print(f"  w {w:<4} w_kn {wk:<4} w_mlm {wm:<5} w_rtd {wr:<5} F1 {f1:.4f}{mark}  "
                  f"({time.time() - t:.0f}s, 누적 {nfw()})", flush=True)

        print("[1단계] 사전학습 모델 가중치 조합 (12b 항은 고정)")
        for wm in mlm_ws:
            for wr in rtd_ws:
                trial(w0, wk0, wm, wr)
        _, _, wm, wr = best[1]
        if wm or wr:
            print(f"[2단계] 최고 조합 (w_mlm {wm}, w_rtd {wr}) 에서 12b 항을 빼 보기")
            for w, wk in ((0.0, wk0), (w0, 0.0), (0.0, 0.0)):
                trial(w, wk, wm, wr)
        f1, (w, wk, wm, wr), preds = best
        print(f"best F1 {f1:.4f} (w {w}, w_kn {wk}, w_mlm {wm}, w_rtd {wr}) | 01 대비 {f1 - base:+.4f} "
              f"| 12b (외부 모델 없음) {b['f1']:.4f}, 09 (Qwen) 0.9908, 11b (Qwen) 0.9915")
        df.assign(pred=preds, f1=[nc.f1_score(p, g) for p, g in zip(preds, gts)]).to_csv(
            os.path.join(o, f"val_pred_anchor_{out_tag}.csv"), index=False, encoding="utf-8-sig")
        if not args.val_limit:
            with open(bpath, "w", encoding="utf-8") as fp:
                json.dump({**{k: getattr(args, k) for k in ("anchor", "min_len", "refs", "nbest", "order", "passes",
                                                             "ctx", "rtd_pad")},
                           "mlm": mlm_name, "w": w, "w_kn": wk, "w_mlm": wm, "w_rtd": wr, "f1": f1, "base_f1": base,
                           "oracle_f1": orc, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, ensure_ascii=False,
                          indent=1)
    else:
        args.rtd_pad = b.get("rtd_pad", args.rtd_pad)
        preds = resolve(p01s, items, sc, {"nn": b["w"], "kn": b["w_kn"], "mlm": b["w_mlm"], "rtd": b["w_rtd"]}, args)
        print(f"  w {b['w']}, w_kn {b['w_kn']}, w_mlm {b['w_mlm']}, w_rtd {b['w_rtd']} (val F1 {b['f1']:.4f}) "
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
    ap.add_argument("--base", default="12b", help="글자 언어모델·5-gram 가중치를 가져올 12 단계 결과의 꼬리표")
    ap.add_argument("--mlm", default="", help="generator 경로 (기본: outputs/kochar_gen_split|full, 미세조정한 것)")
    ap.add_argument("--mlm_weights", default="0,0.25,0.5,1.0")
    ap.add_argument("--rtd_weights", default="0,0.1,0.25,0.5")
    ap.add_argument("--rtd_pad", type=int, default=3, help="discriminator 점수에 넣을 후보 앞뒤 글자 수")
    ap.add_argument("--batch_tokens", type=int, default=65536)
    ap.add_argument("--name", default="13", help="결과 파일 꼬리표")
    ap.add_argument("--unanchor_oov", action="store_true")
    ap.add_argument("--cand_beam", type=int, default=0)
    args = ap.parse_args()
    args.refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    print(vars(args))
    run(args)


if __name__ == "__main__":
    main()
