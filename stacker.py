"""
난독화 한글 리뷰 복원 - 6단계: 스태킹 (01 노이즈 채널 예측 + 사전 특징을 입력으로 받는 태거)

04 처럼 점수를 고정 비율로 더하면 실패했고(태거가 틀릴 때도 과신), 05 에서 사전을 '입력'으로 넣자 모델이
언제 따를지를 배워 순이익이 났다. 같은 방식으로 01 의 예측 글자를 입력으로 넣어, 태거가 01 의 답을
출발점으로 삼고 고칠 곳만 배우게 한다. (01 단독 0.9834, 행별로 01·v2 중 나은 쪽을 고르면 0.9870)

위치별 입력 = 글자·자모 + 사전 특징 5개(05 와 동일) + 01 예측 글자의 음절·초·중·종성 4개

주의 1. 정답 누출: 학습 문장의 01 예측은 그 문장을 본 01 로 만들면 검증·test 보다 훨씬 정확해져서
         모델이 01 을 베끼게 된다 → 5분할 교차 예측(out-of-fold): 01 을 나머지 4/5 로 만들어 1/5 을 예측.
주의 2. 비용: 01 디코딩은 CPU 라 느리다 → 실제 입력 + pool 증강 nc_pool 벌만 미리 디코딩해 캐시.
         새 노이즈 샘플(fresh)은 01·사전 특징 없이(0) 넣는다.

사용 예 (Colab):
    !python -u stacker.py --mode prep        # 01 교차 예측 캐시 (v2 사전 캐시가 없으면 먼저 생성)
    !python -u stacker.py --mode train       # 학습, val 로 best 저장
    !python -u stacker.py --mode predict     # test 예측
"""
import argparse
import json
import math
import os
import pickle
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import noisy_channel as nc
import tagger_v2 as v2
from neural_tagger import (
    N_HANGUL, PAD, CandidateDecoder, CharVocab, NoiseSampler, build_masks, encode, masked_logits, split_train,
)

N_DICT = v2.N_FEAT       # 사전 특징 5개
N_NC = 4                 # 01 예측: 음절, 초+1, 중+1, 종+1 (0 = 정보 없음)
N_IN = 4 + N_DICT + N_NC  # 입력 텐서 수 (그 뒤는 타깃 3개)


# --------------------------------------------------------------------------- #
# 01 예측 특징
# --------------------------------------------------------------------------- #
def nc_params(out_dir):
    """01 sweep 최고 설정 (outputs/best_params.json). 없으면 sweep 최고값으로 알려진 기본값."""
    p = {"order": 5, "cand_thr": 0.005, "no_pair": False, "dict_min_len": 2, "dict_oov": 0.5,
         "beam": 16, "topk": 12, "lm_weight": 1.1, "dict_weight": 3.0}
    path = os.path.join(out_dir, "best_params.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            p.update(json.load(f)["params"])
    return p


def nc_feature(pred, x):
    f = np.zeros((N_NC, len(x)), np.int16)
    for i, (xc, yc) in enumerate(zip(x, pred)):
        if nc.is_hangul(xc) and nc.is_hangul(yc):
            a, b, g = nc.dec(yc)
            f[:, i] = (1 + ord(yc) - 0xAC00, a + 1, b + 1, g + 1)
    return f


def nc_decode(train_df, xs, p, workers, tag):
    t = time.time()
    decoder, _ = nc.build(train_df, argparse.Namespace(**p))
    preds = nc.decode_all(decoder, list(xs), workers)
    print(f"    [{tag}] 01 디코딩 {len(xs)}문장 {time.time() - t:.0f}s", flush=True)
    return preds


def nc_path(args):
    return os.path.join(args.out_dir, f"v3_nc_{'full' if args.full else 'split'}.pkl")


def prep(args):
    v2path = v2.cache_path(args)
    if not os.path.exists(v2path):
        print(f"  {v2path} 가 없어 사전 특징 캐시부터 만듭니다 (tagger_v2 prep)")
        v2.prep(args)
    if os.path.exists(nc_path(args)) and not args.force:
        print(f"  {nc_path(args)} 가 이미 있어 01 교차 예측은 건너뜁니다 (다시 만들려면 --force)")
        return
    with open(v2path, "rb") as fp:
        cache = pickle.load(fp)
    items, pool = cache["items"], cache["pool"]

    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    tr, va = (full, None) if args.full else split_train(full, args.val_ratio, args.seed)
    kept = [i for i, (x, y) in enumerate(zip(tr.input, tr.output)) if len(x.rstrip()) == len(y.rstrip())]
    n = len(items) // (1 + pool)
    assert n == len(kept), f"v2 캐시 문장 수({n})와 분할({len(kept)})이 다릅니다. v2 prep 을 다시 실행하세요."
    p = nc_params(args.out_dir)
    nv = min(args.nc_pool, pool)
    print(f"  01 설정: {p}")
    print(f"  5분할 교차 예측: 문장 {n} x (실제 1 + 증강 {nv}) = {n * (1 + nv)} 디코딩, workers {args.workers}")

    feats, real_pred = [None] * len(items), {}
    t0 = time.time()
    for k in range(args.folds):
        fold_s = [s for s in range(n) if kept[s] % args.folds == k]
        fold_tr = tr.iloc[[i for i in range(len(tr)) if i % args.folds != k]]
        idx = [s * (1 + pool) + v for s in fold_s for v in range(nv + 1)]
        preds = nc_decode(fold_tr, [items[i][0] for i in idx], p, args.workers, f"fold {k + 1}/{args.folds}")
        for i, pr in zip(idx, preds):
            feats[i] = nc_feature(pr, items[i][0])
            if i % (1 + pool) == 0:
                real_pred[i] = pr
    oof = nc.mean_f1([real_pred[s * (1 + pool)] for s in range(n)], [items[s * (1 + pool)][1] for s in range(n)])
    print(f"  교차 예측 완료 {time.time() - t0:.0f}s | 학습 문장(실제 입력)에서 01 F1 {oof:.4f}")

    out = {"nc": feats, "nc_pool": nv, "folds": args.folds, "params": p, "oof_f1": oof}
    if va is not None:
        vp = nc_decode(tr, va.input, p, args.workers, "val")
        out["val_nc"] = [nc_feature(pr, x) for pr, x in zip(vp, va.input)]
        out["val_f1_01"] = nc.mean_f1(vp, list(va.output))
        print(f"  val 에서 01 F1 {out['val_f1_01']:.4f}  (학습 문장 교차 예측 {oof:.4f} 와 비슷해야 정상)")
    with open(nc_path(args), "wb") as fp:
        pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved {nc_path(args)}")


# --------------------------------------------------------------------------- #
# 모델·데이터
# --------------------------------------------------------------------------- #
class StackTagger(v2.DictTagger):
    """DictTagger + 01 예측 임베딩. 01 임베딩은 0 으로 시작 → v2·02 체크포인트에서 이어 학습 가능."""

    def __init__(self, n_char, d=384, **kw):
        super().__init__(n_char, d=d, **kw)
        self.n_syl = nn.Embedding(N_HANGUL + 1, d, padding_idx=0)
        self.n_cho = nn.Embedding(nc.N_CHO + 1, d, padding_idx=0)
        self.n_jung = nn.Embedding(nc.N_JUNG + 1, d, padding_idx=0)
        self.n_jong = nn.Embedding(nc.N_JONG + 1, d, padding_idx=0)
        for e in (self.n_syl, self.n_cho, self.n_jung, self.n_jong):
            nn.init.zeros_(e.weight)

    def forward(self, c, a, b, g, d1, p1, d2, p2, po, n1, n2, n3, n4):
        pos = torch.arange(c.size(1), device=c.device)
        h = self.char(c) + self.cho(a) + self.jung(b) + self.jong(g) + self.pos(pos)[None]
        h = h + self.d1(d1) + self.p1(p1) + self.d2(d2) + self.p2(p2) + self.po(po)
        h = h + self.n_syl(n1) + self.n_cho(n2) + self.n_jung(n3) + self.n_jong(n4)
        h = self.enc(self.drop(h), src_key_padding_mask=c.eq(PAD))
        h = self.norm(h)
        return self.h_cho(h), self.h_jung(h), self.h_jong(h)


def make_model(vocab, a, device):
    return StackTagger(vocab.size, d=a.d, layers=a.layers, heads=a.heads, ff=a.d * 4,
                       drop=a.dropout, max_pos=max(512, a.max_len)).to(device)


def collate(batch):
    n = max(len(b[0]) for b in batch)
    out = []
    for k in range(len(batch[0])):
        arr = np.full((len(batch), n), -100 if k >= N_IN else 0, np.int64)
        for i, b in enumerate(batch):
            arr[i, : len(b[k])] = b[k]
        out.append(torch.from_numpy(arr))
    return out


class StackSet(Dataset):
    """epoch 마다 문장당: 실제 입력 1 + 증강 aug 개 + 새 노이즈 fresh 개 (05 와 같은 구성).
    증강은 절반 확률로 01 예측이 있는 변형(1..nc_pool)에서 뽑는다. feat_drop·nc_drop 은 특징을 통째로 끌 확률."""

    def __init__(self, cache, ncc, vocab, args, sampler):
        self.items, self.feats, self.pool = cache["items"], cache["feats"], cache["pool"]
        self.nc, self.nc_pool = ncc["nc"], ncc["nc_pool"]
        self.n = len(self.items) // (1 + self.pool)
        self.vocab, self.max_len, self.sampler = vocab, args.max_len, sampler
        self.aug, self.fresh = min(args.aug, self.pool), args.fresh if sampler else 0
        self.feat_drop, self.nc_drop = args.feat_drop, args.nc_drop

    def __len__(self):
        return self.n * (1 + self.aug + self.fresh)

    def __getitem__(self, k):
        s, r = k % self.n, k // self.n
        if r > self.aug:  # 새 노이즈: 사전·01 특징 없음
            _, y = self.items[s * (1 + self.pool)]
            x = self.sampler.corrupt(y)
            fd = np.zeros((N_DICT, len(x)), np.int16)
            fn = np.zeros((N_NC, len(x)), np.int16)
        else:
            if r == 0:
                v = 0
            elif self.nc_pool and random.random() < 0.5:
                v = random.randint(1, self.nc_pool)
            else:
                v = random.randint(1, self.pool)
            idx = s * (1 + self.pool) + v
            (x, y), fd = self.items[idx], self.feats[idx]
            fn = self.nc[idx] if self.nc[idx] is not None else np.zeros((N_NC, len(x)), np.int16)
            if self.feat_drop and random.random() < self.feat_drop:
                fd = np.zeros_like(fd)
            if self.nc_drop and random.random() < self.nc_drop:
                fn = np.zeros_like(fn)
        if len(x) > self.max_len:
            st = random.randint(0, len(x) - self.max_len)
            x, y = x[st : st + self.max_len], y[st : st + self.max_len]
            fd, fn = fd[:, st : st + self.max_len], fn[:, st : st + self.max_len]
        c, a, b, g, ta, tb, tg = encode(self.vocab, x, y)
        return (c, a, b, g, *[fd[j].astype(np.int64) for j in range(N_DICT)],
                *[fn[j].astype(np.int64) for j in range(N_NC)], ta, tb, tg)


@torch.no_grad()
def predict_logprobs(model, vocab, masks, texts, dfeats, nfeats, device, win=256, batch=64):
    model.eval()
    jobs = []
    for k, x in enumerate(texts):
        if len(x) <= win:
            jobs.append((k, 0, len(x)))
        else:
            starts = list(range(0, len(x) - win + 1, win // 2))
            if starts[-1] != len(x) - win:
                starts.append(len(x) - win)
            jobs += [(k, s, s + win) for s in starts]
    acc = [[np.zeros((len(x), n), np.float32) for n in (nc.N_CHO, nc.N_JUNG, nc.N_JONG)] for x in texts]
    cnt = [np.zeros(len(x), np.float32) for x in texts]
    gm = [m.to(device) for m in masks]
    for i in range(0, len(jobs), batch):
        chunk = jobs[i : i + batch]
        items = [(*encode(vocab, texts[k][s:e]),
                  *[dfeats[k][j, s:e].astype(np.int64) for j in range(N_DICT)],
                  *[nfeats[k][j, s:e].astype(np.int64) for j in range(N_NC)]) for k, s, e in chunk]
        t = [x.to(device) for x in collate(items)]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outs = model(*t)
        lps = [F.log_softmax(masked_logits(o, ids, m), -1).cpu().numpy() for o, ids, m in zip(outs, t[1:4], gm)]
        for j, (k, s, e) in enumerate(chunk):
            for h in range(3):
                acc[k][h][s:e] += lps[h][j, : e - s]
            cnt[k][s:e] += 1
    return [[h / np.maximum(cn, 1)[:, None] for h in a3] for a3, cn in zip(acc, cnt)]


def log_epoch(out_dir, row):
    path = os.path.join(out_dir, "stack_runs.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


HP_KEYS = ["run_name", "full", "folds", "nc_pool", "aug", "fresh", "feat_drop", "nc_drop", "max_len", "d", "layers",
           "heads", "dropout", "epochs", "batch", "lr", "seed", "init_ckpt"]


def train(args, device):
    for path, mode in ((v2.cache_path(args), "prep"), (nc_path(args), "prep")):
        if not os.path.exists(path):
            raise SystemExit(f"{path} 가 없습니다. 먼저 --mode prep {'--full ' if args.full else ''}을 실행하세요.")
    with open(v2.cache_path(args), "rb") as fp:
        cache = pickle.load(fp)
    with open(nc_path(args), "rb") as fp:
        ncc = pickle.load(fp)
    ca = cache["args"]
    for k in ("dict_min_len", "dict_oov", "cand_thr", "pool"):
        setattr(args, k, ca[k])
    args.nc_pool, args.folds = ncc["nc_pool"], ncc["folds"]
    print(f"  01 교차 예측 F1 {ncc['oof_f1']:.4f}" + (f" | val 01 F1 {ncc['val_f1_01']:.4f}" if "val_f1_01" in ncc else ""))

    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    tr, va = (full, None) if args.full else split_train(full, args.val_ratio, args.seed)
    vocab = CharVocab(list(full.input) + list(full.output) + list(test.input))
    ch = nc.JamoChannel(nc.aligned_rows(tr), cand_thr=args.cand_thr)
    masks = build_masks(ch)
    gmasks = [m.to(device) for m in masks]
    dec_ = CandidateDecoder(ch, {c for t in tr.output for c in t if nc.is_hangul(c)})

    model = make_model(vocab, args, device)
    if args.init_ckpt:
        st = torch.load(args.init_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(st["model"], strict=False)
        print(f"  init from {args.init_ckpt} (epoch {st['epoch']}) | 새로 추가된 층 {len(missing)}개, 무시 {len(unexpected)}개")

    sampler = NoiseSampler(nc.JamoChannel(nc.aligned_rows(tr))) if args.fresh else None
    ds = StackSet(cache, ncc, vocab, args, sampler)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=args.loader_workers,
                    drop_last=True, persistent_workers=args.loader_workers > 0)
    print(f"  params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | 문장 {ds.n} x (실제 1 + 증강 {ds.aug}"
          f" + 새 노이즈 {ds.fresh}) | 끄기: 사전 {ds.feat_drop:.0%} 01 {ds.nc_drop:.0%} | steps/epoch {len(dl)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(dl)
    warm = int(0.05 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sfx = f"_{args.run_name}" if args.run_name else ""
    ckpt = os.path.join(args.out_dir, f"stack_full{sfx}.pt" if args.full else f"stack_best{sfx}.pt")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    hp = {k: getattr(args, k) for k in HP_KEYS}
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for batch in dl:
            batch = [t.to(device, non_blocking=True) for t in batch]
            inp, tgt = batch[:N_IN], batch[N_IN:]
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                outs = model(*inp)
            loss = sum(F.cross_entropy(masked_logits(o, ids, m, t).transpose(1, 2), t, ignore_index=-100)
                       for o, ids, m, t in zip(outs, inp[1:4], gmasks, tgt))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        sec = time.time() - t0
        msg = f"epoch {ep:3d} | loss {tot / len(dl):.4f} | {sec:.0f}s"
        state = {"model": model.state_dict(), "args": vars(args), "epoch": ep, "nc_params": ncc["params"]}
        f1 = None
        if va is not None and (ep % args.eval_every == 0 or ep == args.epochs):
            lps = predict_logprobs(model, vocab, masks, list(va.input), cache["val_feats"], ncc["val_nc"], device,
                                   win=args.max_len)
            preds = [dec_.decode(x, lp) for x, lp in zip(va.input, lps)]
            f1 = nc.mean_f1(preds, list(va.output))
            msg += f" | val F1 {f1:.4f}"
            if f1 > best:
                best = f1
                torch.save(state, ckpt)
                va.assign(pred=preds, f1=[nc.f1_score(p, g) for p, g in zip(preds, va.output)]).to_csv(
                    os.path.join(args.out_dir, f"val_pred_stack{sfx}.csv"), index=False, encoding="utf-8-sig")
                msg += " *"
        elif va is None:
            torch.save(state, ckpt)
        print(msg, flush=True)
        log_epoch(args.out_dir, {"run_id": run_id, "epoch": ep, "loss": round(tot / len(dl), 5),
                                 "val_f1": None if f1 is None else round(f1, 5),
                                 "n_val": 0 if va is None else len(va), "sec": round(sec, 1), **hp})
    if va is not None:
        print(f"best val F1 {best:.4f} -> {ckpt}  (01 단독 {ncc.get('val_f1_01', float('nan')):.4f}, "
              f"v2 태거 0.9758 과 비교)")


def predict(args, device):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    sfx = f"_{args.run_name}" if args.run_name else ""
    ckpt = args.ckpt or os.path.join(args.out_dir, f"stack_best{sfx}.pt")
    st = torch.load(ckpt, map_location="cpu")
    a = argparse.Namespace(**st["args"])
    print(f"loaded {ckpt} (epoch {st['epoch']})")
    # CPU 병렬 작업(사전 특징, 01 디코딩)을 GPU 사용 전에
    ch = nc.JamoChannel(nc.aligned_rows(full), cand_thr=a.cand_thr)
    wd, vocab_chars = v2.build_dict(full, ch, a)
    dfeats = v2.compute_feats(wd, list(test.input), [None] * len(test), args.workers)
    preds01 = nc_decode(full, test.input, st["nc_params"], args.workers, "test")
    nfeats = [nc_feature(pr, x) for pr, x in zip(preds01, test.input)]
    vocab = CharVocab(list(full.input) + list(full.output) + list(test.input))
    model = make_model(vocab, a, device)
    model.load_state_dict(st["model"])
    lps = predict_logprobs(model, vocab, build_masks(ch), list(test.input), dfeats, nfeats, device, win=a.max_len)
    dec_ = CandidateDecoder(ch, vocab_chars)
    preds = [dec_.decode(x, lp) for x, lp in zip(test.input, lps)]
    changed = sum(p != q for p, q in zip(preds, preds01))
    print(f"  01 예측과 다른 문장 {changed}/{len(preds)}")
    sub["output"] = sub.ID.map(dict(zip(test.ID, preds)))
    out = os.path.join(args.out_dir, f"submission_stack{sfx}.csv")
    sub.to_csv(out, index=False, encoding="utf-8-sig")
    with open(os.path.join(args.out_dir, f"submission_stack{sfx}_params.json"), "w", encoding="utf-8") as fp:
        json.dump({"ckpt": ckpt, "epoch": st["epoch"], "train_args": st["args"], "nc_params": st["nc_params"],
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, ensure_ascii=False, indent=1)
    print(f"saved {out} (+ _params.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prep", "train", "predict"], default="train")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--run_name", default="", help="결과 파일 꼬리표 (stack_best_<이름>.pt 등)")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    # prep (01 교차 예측 / v2 사전 특징 — v2 캐시가 없을 때만 사용)
    ap.add_argument("--force", action="store_true", help="prep: 01 교차 예측 캐시가 있어도 다시 만들기")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--nc_pool", type=int, default=1, help="01 로 미리 디코딩할 증강 변형 수 (실제 입력은 항상)")
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--dict_min_len", type=int, default=2)
    ap.add_argument("--dict_oov", type=float, default=0.5)
    ap.add_argument("--cand_thr", type=float, default=0.002)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    # 학습
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--fresh", type=int, default=2)
    ap.add_argument("--feat_drop", type=float, default=0.3)
    ap.add_argument("--nc_drop", type=float, default=0.2, help="01 특징을 통째로 끌 확률 (01 베끼기 방지)")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--init_ckpt", default="")
    ap.add_argument("--loader_workers", type=int, default=2)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(vars(args), "| device", device)
    {"prep": lambda: prep(args), "train": lambda: train(args, device), "predict": lambda: predict(args, device)}[
        args.mode]()


if __name__ == "__main__":
    main()
