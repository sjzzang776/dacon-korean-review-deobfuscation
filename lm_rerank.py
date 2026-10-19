"""
난독화 한글 리뷰 복원 - 7단계: 사전학습 언어모델(공개 가중치)로 01 후보 재순위화

지금 모델들은 train 정답 1만 문장으로만 한국어를 배워서, train 에 없는 단어(검증 어절의 약 12.5%:
소탐대실·개막장·허난설헌 등)를 모른다. 대회 규칙상 외부 '데이터'는 금지지만 가중치가 공개된 사전학습
모델은 허용되므로, 대량의 한국어를 이미 배운 공개 언어모델로 01 이 낸 후보 중 자연스러운 것을 고른다.

  1. 문장을 chunk_words 어절씩 조각으로 나눔 (문장 전체 후보는 서로 거의 같아서 조각 단위로)
  2. 조각마다 01 이 상위 nbest 개 후보 + 점수를 냄 (앞 조각의 01 결과를 문맥으로 이어 받음)
  3. 언어모델이 '앞 문맥(01 결과) + 후보' 의 log P(후보 | 앞 문맥) 계산
  4. 조각마다 01 점수 + lm_weight * LM 점수 가 최대인 후보 선택. lm_weight 는 val 로 탐색 (0 = 01 단독)

캐시: outputs/rerank_cands_{val|test}.pkl (01 후보, CPU), outputs/rerank_lm_{val|test}_{모델}.pkl (LM 점수, GPU)
→ 가중치 탐색은 캐시만 다시 조합하므로 순식간.

사용 예 (Colab, GPU):
    !python -u lm_rerank.py --mode val                       # 후보 생성 → LM 점수 → 가중치 탐색
    !python -u lm_rerank.py --mode test                      # best 가중치로 제출 파일
    !python -u lm_rerank.py --mode val --lm Qwen/Qwen2.5-0.5B  # 다른 공개 모델
"""
import argparse
import json
import os
import pickle
import re
import time
from multiprocessing import Pool

import pandas as pd

import noisy_channel as nc
from neural_tagger import split_train


# --------------------------------------------------------------------------- #
# 1. 01 후보 생성 (CPU)
# --------------------------------------------------------------------------- #
def chunks_of(x, words):
    """어절(공백 포함) words 개씩 묶은 조각. ''.join(조각) == x."""
    pieces = re.findall(r"\S+\s*", x)
    if not pieces:
        return [x] if x else []
    pieces[0] = x[: len(x) - len(x.lstrip())] + pieces[0]
    out = ["".join(pieces[i : i + words]) for i in range(0, len(pieces), words)]
    assert "".join(out) == x
    return out


_G = {}


def _nbest_sentence(x):
    dec, k, words = _G["dec"], _G["nbest"], _G["words"]
    out, ctx_y, ctx_x = [], "", ""
    chs = chunks_of(x, words)
    for j, cx in enumerate(chs):
        cands = dec.decode_nbest(cx, k, ctx_y=ctx_y, ctx_x=ctx_x, final=(j == len(chs) - 1))
        out.append({"x": cx, "prefix": ctx_y, "cands": [[y, s] for y, s in cands]})
        ctx_y += cands[0][0]
        ctx_x += cx
    return out


def nc_params(out_dir):
    p = {"order": 5, "cand_thr": 0.005, "no_pair": False, "dict_min_len": 2, "dict_oov": 0.5,
         "beam": 16, "topk": 12, "lm_weight": 1.1, "dict_weight": 3.0}
    path = os.path.join(out_dir, "best_params.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            p.update(json.load(f)["params"])
    return p


def make_cands(train_df, texts, args):
    p = nc_params(args.out_dir)
    print(f"  01 설정: {p} | 조각 {args.chunk_words}어절, 후보 {args.nbest}개")
    dec, _ = nc.build(train_df, argparse.Namespace(**p))
    _G.update(dec=dec, nbest=args.nbest, words=args.chunk_words)
    t = time.time()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            res = list(pool.imap(_nbest_sentence, list(texts), chunksize=8))
    else:
        res = [_nbest_sentence(x) for x in texts]
    n_chunk = sum(len(r) for r in res)
    n_cand = sum(len(c["cands"]) for r in res for c in r)
    print(f"  후보 생성 {len(res)}문장, 조각 {n_chunk}개, 후보 {n_cand}개 ({time.time() - t:.0f}s)")
    return {"sents": res, "nc_params": p, "nbest": args.nbest, "chunk_words": args.chunk_words}


# --------------------------------------------------------------------------- #
# 2. 언어모델 점수 (GPU)
# --------------------------------------------------------------------------- #
def lm_scores(cands, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.lm)
    model = AutoModelForCausalLM.from_pretrained(
        args.lm, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"  LM {args.lm} ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B) on {device}")

    # (문맥 토큰, 후보 토큰) 목록. 문맥은 앞 조각들의 01 결과 마지막 ctx_chars 글자.
    jobs = []
    for si, sent in enumerate(cands["sents"]):
        for ci, ch in enumerate(sent):
            ctx = tok(ch["prefix"][-args.ctx_chars :], add_special_tokens=False).input_ids
            for k, (y, _) in enumerate(ch["cands"]):
                cont = tok(y, add_special_tokens=False).input_ids
                jobs.append(([start] + ctx, cont, (si, ci, k)))
    order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][0]) + len(jobs[j][1]))
    scores = [0.0] * len(jobs)
    t = time.time()
    with torch.no_grad():
        i = 0
        while i < len(order):
            # 토큰 수 기준으로 배치 크기 조절 (길이 오름차순이므로 창의 마지막이 가장 긺)
            last = jobs[order[min(i + 255, len(order) - 1)]]
            bs = max(1, min(256, args.batch_tokens // max(len(last[0]) + len(last[1]), 1)))
            idx = order[i : i + bs]
            seqs = [jobs[j][0] + jobs[j][1] for j in idx]
            n = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), n), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), n), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, : len(s)] = torch.tensor(s)
                att[r, : len(s)] = 1
            ids, att = ids.to(device), att.to(device)
            logits = model(input_ids=ids, attention_mask=att).logits  # (B, T, 어휘) — 어휘가 커서 필요한 위치만 float 변환
            for r, j in enumerate(idx):
                c0, cl = len(jobs[j][0]), len(jobs[j][1])
                if cl == 0:
                    continue
                lp = torch.log_softmax(logits[r, c0 - 1 : c0 + cl - 1].float(), -1)
                scores[j] = float(lp.gather(-1, ids[r, c0 : c0 + cl, None]).sum())
            del logits
            i += bs
    print(f"  LM 점수 {len(jobs)}개 ({time.time() - t:.0f}s)")
    out = [[[0.0] * len(ch["cands"]) for ch in sent] for sent in cands["sents"]]
    for (_, _, (si, ci, k)), s in zip(jobs, scores):
        out[si][ci][k] = s
    return {"lm": args.lm, "scores": out, "ctx_chars": args.ctx_chars}


# --------------------------------------------------------------------------- #
# 3. 조합
# --------------------------------------------------------------------------- #
def assemble(cands, lm, w):
    preds = []
    for sent, sc in zip(cands["sents"], lm["scores"]):
        s = []
        for ch, ls in zip(sent, sc):
            best = max(range(len(ch["cands"])), key=lambda k: ch["cands"][k][1] + w * ls[k])
            s.append(ch["cands"][best][0])
        preds.append("".join(s))
    return preds


def oracle(cands, gts):
    """조각마다 정답과 가장 많이 일치하는 후보를 골랐을 때 (재순위화로 갈 수 있는 상한)."""
    preds = []
    for sent, g in zip(cands["sents"], gts):
        pos, s = 0, []
        for ch in sent:
            seg = g[pos : pos + len(ch["x"])]
            s.append(max((c[0] for c in ch["cands"]), key=lambda y: sum(a == b for a, b in zip(y, seg))))
            pos += len(ch["x"])
        preds.append("".join(s))
    return preds


def lm_tag(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.split("/")[-1])


def load_or(path, make):
    if os.path.exists(path):
        with open(path, "rb") as f:
            print(f"  캐시 사용: {path}")
            return pickle.load(f)
    obj = make()
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved {path}")
    return obj


def log_row(out_dir, row):
    path = os.path.join(out_dir, "rerank_runs.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run(args):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    split = args.mode
    if split == "val":
        tr, df = split_train(full, args.val_ratio, args.seed)
        if args.val_limit:
            df = df.iloc[: args.val_limit]
    else:
        tr, df = full, pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    sfx = f"_n{args.val_limit}" if split == "val" and args.val_limit else ""
    cpath = os.path.join(args.out_dir, f"rerank_cands_{split}{sfx}_c{args.chunk_words}_k{args.nbest}.pkl")
    cands = load_or(cpath, lambda: make_cands(tr, df.input, args))
    lpath = os.path.join(args.out_dir, f"rerank_lm_{split}{sfx}_c{args.chunk_words}_k{args.nbest}_{lm_tag(args.lm)}.pkl")
    lm = load_or(lpath, lambda: lm_scores(cands, args))

    if split == "val":
        gts = list(df.output)
        base = nc.mean_f1(assemble(cands, lm, 0.0), gts)
        orc = nc.mean_f1(oracle(cands, gts), gts)
        print(f"  01 단독(조각 디코딩) F1 {base:.4f} | 후보 중 최선을 고를 때 상한 {orc:.4f}")
        best = (-1, None, None)
        for w in [float(v) for v in args.weights.split(",")]:
            preds = assemble(cands, lm, w)
            f1 = nc.mean_f1(preds, gts)
            log_row(args.out_dir, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "lm": args.lm, "n_val": len(df),
                                   "chunk_words": args.chunk_words, "nbest": args.nbest, "ctx_chars": args.ctx_chars,
                                   "lm_weight_rerank": w, "f1": round(f1, 5), "base_f1": round(base, 5),
                                   "oracle_f1": round(orc, 5)})
            mark = ""
            if f1 > best[0]:
                best, mark = (f1, w, preds), " *"
            print(f"  lm_weight {w:<5} F1 {f1:.4f}{mark}", flush=True)
        f1, w, preds = best
        print(f"best F1 {f1:.4f} (lm_weight {w}) | 01 대비 {f1 - base:+.4f}")
        df.assign(pred=preds, f1=[nc.f1_score(p, g) for p, g in zip(preds, gts)]).to_csv(
            os.path.join(args.out_dir, f"val_pred_rerank_{lm_tag(args.lm)}.csv"), index=False, encoding="utf-8-sig")
        if not args.val_limit:
            with open(os.path.join(args.out_dir, f"best_rerank_{lm_tag(args.lm)}.json"), "w", encoding="utf-8") as fp:
                json.dump({"lm": args.lm, "lm_weight": w, "f1": f1, "base_f1": base, "oracle_f1": orc,
                           "chunk_words": args.chunk_words, "nbest": args.nbest, "ctx_chars": args.ctx_chars,
                           "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, ensure_ascii=False, indent=1)
    else:
        bpath = os.path.join(args.out_dir, f"best_rerank_{lm_tag(args.lm)}.json")
        if not os.path.exists(bpath):
            raise SystemExit(f"{bpath} 가 없습니다. 먼저 --mode val (val_limit 없이) 을 실행하세요.")
        with open(bpath, encoding="utf-8") as fp:
            best = json.load(fp)
        preds = assemble(cands, lm, best["lm_weight"])
        base = assemble(cands, lm, 0.0)
        print(f"  lm_weight {best['lm_weight']} (val F1 {best['f1']:.4f}) | 01 과 다른 문장 "
              f"{sum(a != b for a, b in zip(preds, base))}/{len(preds)}")
        sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
        sub["output"] = sub.ID.map(dict(zip(df.ID, preds)))
        out = os.path.join(args.out_dir, f"submission_rerank_{lm_tag(args.lm)}.csv")
        sub.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "test"], default="val")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_limit", type=int, default=0, help="빠른 확인용 검증 행 수 (0 = 전체)")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-1.5B", help="공개 가중치 causal LM (Hugging Face 이름)")
    ap.add_argument("--chunk_words", type=int, default=3, help="조각당 어절 수")
    ap.add_argument("--nbest", type=int, default=8, help="조각당 01 후보 수 (<= 01 beam)")
    ap.add_argument("--ctx_chars", type=int, default=150, help="LM 에 줄 앞 문맥 글자 수")
    ap.add_argument("--batch_tokens", type=int, default=4096, help="LM 배치당 토큰 수 (GPU 메모리 부족 시 줄이기)")
    ap.add_argument("--weights", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5", help="탐색할 LM 가중치")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    print(vars(args))
    os.makedirs(args.out_dir, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
