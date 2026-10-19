"""
난독화 한글 리뷰 복원 - 5단계: 어절 사전을 인코더 입력으로 넣은 태거 (tagger v2)

03·04 에서는 사전 점수를 '고정 가중치로 더하기'만 했다. 그러면 사전이 '누수'를 1순위로 줘도
문맥상 '뉴스'가 맞는 경우를 모델이 가려낼 수 없다. v2 는 사전의 제안을 입력 특징으로 넣어서,
모델이 문맥을 보고 언제 사전을 따르고 언제 무시할지를 학습한다.

위치별 사전 특징 (어절 단위로 계산해 글자 위치에 펼침):
  - 사전 1순위 글자 + 그 확률 구간, 2순위 글자 + 확률 구간
  - '사전에 없음(OOV)' 확률 구간
  - 2글자 미만 어절·사전 미적용 위치는 0 (= 정보 없음)

주의 1. 정답 누출: 학습 문장의 어절은 그 문장으로 만든 사전에 항상 들어 있다.
         -> 학습 문장의 특징은 '자기 문장을 뺀 사전'(leave-one-out) 으로 계산.
주의 2. 속도: 사전 사후확률은 파이썬이라 느리다.
         -> 증강 문장을 미리 pool 벌 만들어 두고 특징을 한 번만 계산해 저장(prep 단계).

사용 예 (Colab):
    !python tagger_v2.py --mode prep                 # 특징 캐시 생성 (CPU, 수 분)
    !python tagger_v2.py --mode train                # 학습 (GPU), val 로 best 저장
    !python tagger_v2.py --mode predict              # test 예측
    # 02 체크포인트에서 이어 학습: --init_ckpt outputs/tagger_best.pt --epochs 15
"""
import argparse
import copy
import json
import math
import os
import pickle
import random
import time
import warnings
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import noisy_channel as nc
from neural_tagger import (
    N_HANGUL, PAD, CandidateDecoder, CharVocab, NoiseSampler, Tagger, build_masks, encode, masked_logits,
    split_train,
)

N_BUCKET = 12  # 확률 구간 id: 0 = 정보 없음, 1..11 = [0,0.1) ... [1.0]
N_FEAT = 5     # d1, p1, d2, p2, oov


def bucket(p):
    return 1 + min(int(p * 10), 10)


# --------------------------------------------------------------------------- #
# 사전 특징
# --------------------------------------------------------------------------- #
def word_posterior(wd, xr, own=None):
    """입력 어절 xr 에 대한 사전 어절 사후확률 [(w, p)] 과 OOV 확률. own: 자기 문장 어절 빈도(leave-one-out)."""
    cands = wd.index.get(wd.key(xr))
    if not cands:
        return None, 1.0
    scored = []
    for c, w in cands:
        if own:
            c -= own.get(w, 0)
        if c > 0:
            scored.append((math.log(c) + wd.ch.seq_logp(xr, w), w))
    if not scored:
        return None, 1.0
    oov = wd.log_oov + sum(wd._best_char_logp(xc) for xc in xr)
    mx = max(oov, max(s for s, _ in scored))
    z = math.exp(oov - mx) + sum(math.exp(s - mx) for s, _ in scored)
    return [(w, math.exp(s - mx) / z) for s, w in scored], math.exp(oov - mx) / z


def dict_features(wd, x, own=None):
    """x 의 위치별 (d1, p1, d2, p2, oov) int16 배열 5개. 사전 글자 id = 1 + (음절 - '가'), 0 = 없음."""
    n = len(x)
    f = np.zeros((N_FEAT, n), np.int16)
    for i0, xr in nc.hangul_runs(x):
        if len(xr) < wd.min_len:
            continue
        post, p_oov = word_posterior(wd, xr, own)
        f[4, i0 : i0 + len(xr)] = bucket(p_oov)
        if not post:
            continue
        for j in range(len(xr)):
            acc = Counter()
            for w, p in post:
                acc[w[j]] += p
            top = acc.most_common(2)
            f[0, i0 + j] = 1 + ord(top[0][0]) - 0xAC00
            f[1, i0 + j] = bucket(top[0][1])
            if len(top) > 1:
                f[2, i0 + j] = 1 + ord(top[1][0]) - 0xAC00
                f[3, i0 + j] = bucket(top[1][1])
    return f


_G = {}


def _feat_job(item):
    x, own = item
    return dict_features(_G["wd"], x, own)


def compute_feats(wd, xs, owns, workers):
    _G["wd"] = wd
    items = list(zip(xs, owns))
    out, t0, step = [], time.time(), max(1, len(items) // 10)
    pool = Pool(workers) if workers > 1 else None
    try:
        it = pool.imap(_feat_job, items, chunksize=64) if pool else map(_feat_job, items)
        for k, f in enumerate(it, 1):
            out.append(f)
            if k % step == 0 or k == len(items):
                print(f"    특징 {k}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    return out


def own_counts(y, min_len):
    return Counter(w for _, w in nc.hangul_runs(y) if len(w) >= min_len)


def build_dict(df, ch, args):
    vocab_chars = {c for t in df.output for c in t if nc.is_hangul(c)}
    wd = nc.WordDict(df.output, ch, vocab_chars, min_len=args.dict_min_len, oov_count=args.dict_oov)
    return wd, vocab_chars


def feat_report(name, xs, ys, feats):
    """특징 진단: 한글 위치 중 사전 제안 비율, 제안 1순위 정확도, 평균 OOV 구간."""
    n = cov = hit = 0
    oov = []
    for x, y, f in zip(xs, ys, feats):
        for i, (xc, yc) in enumerate(zip(x, y)):
            if not nc.is_hangul(xc):
                continue
            n += 1
            if f[4, i]:
                oov.append(f[4, i])
            if f[0, i]:
                cov += 1
                hit += chr(0xAC00 + int(f[0, i]) - 1) == yc
    print(f"  [{name}] 사전 제안 위치 {cov / max(n, 1):.3f} | 제안 1순위 정확도 {hit / max(cov, 1):.3f} | "
          f"평균 OOV 구간 {np.mean(oov) if oov else 0:.2f}")


def cache_path(args):
    return os.path.join(args.out_dir, f"v2_cache_{'full' if args.full else 'split'}.pkl")


def prep(args):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    tr, va = (full, None) if args.full else split_train(full, args.val_ratio, args.seed)
    rows = nc.aligned_rows(tr)
    ch = nc.JamoChannel(rows, cand_thr=args.cand_thr)
    wd, _ = build_dict(tr, ch, args)
    sampler = NoiseSampler(ch)
    random.seed(args.seed)
    np.random.seed(args.seed)

    t = time.time()
    items = []  # 문장 s 의 변형 v 는 인덱스 s*(pool+1)+v (v=0 실제 입력, 1.. 증강)
    for x, y in rows:
        items.append((x, y))
        for _ in range(args.pool):
            items.append((sampler.corrupt(y), y))
    owns = [own_counts(y, args.dict_min_len) for _, y in items]
    feats = compute_feats(wd, [x for x, _ in items], owns, args.workers)
    print(f"  train 특징 {len(items)}개 (문장 {len(rows)} x {1 + args.pool}) {time.time() - t:.0f}s")
    real = slice(0, None, 1 + args.pool)
    feat_report("train 실제 입력, leave-one-out", [x for x, _ in items[real]], [y for _, y in items[real]], feats[real])

    cache = {"items": items, "feats": feats, "pool": args.pool, "args": vars(args)}
    if va is not None:
        t = time.time()
        vf = compute_feats(wd, list(va.input), [None] * len(va), args.workers)
        print(f"  val 특징 {len(va)}개 {time.time() - t:.0f}s")
        feat_report("val", list(va.input), list(va.output), vf)
        cache["val_feats"] = vf
    os.makedirs(args.out_dir, exist_ok=True)
    with open(cache_path(args), "wb") as fp:
        pickle.dump(cache, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved {cache_path(args)}")


# --------------------------------------------------------------------------- #
# 모델
# --------------------------------------------------------------------------- #
class DictTagger(Tagger):
    """Tagger + 사전 특징 임베딩. 사전 임베딩은 0 으로 시작 → 02 체크포인트에서 이어 학습해도 처음엔 02 와 동일."""

    def __init__(self, n_char, d=384, **kw):
        super().__init__(n_char, d=d, **kw)
        self.d1 = nn.Embedding(N_HANGUL + 1, d, padding_idx=0)
        self.d2 = nn.Embedding(N_HANGUL + 1, d, padding_idx=0)
        self.p1 = nn.Embedding(N_BUCKET, d, padding_idx=0)
        self.p2 = nn.Embedding(N_BUCKET, d, padding_idx=0)
        self.po = nn.Embedding(N_BUCKET, d, padding_idx=0)
        for e in (self.d1, self.d2, self.p1, self.p2, self.po):
            nn.init.zeros_(e.weight)

    def forward(self, c, a, b, g, d1, p1, d2, p2, po):
        pos = torch.arange(c.size(1), device=c.device)
        h = self.char(c) + self.cho(a) + self.jung(b) + self.jong(g) + self.pos(pos)[None]
        h = h + self.d1(d1) + self.p1(p1) + self.d2(d2) + self.p2(p2) + self.po(po)
        h = self.enc(self.drop(h), src_key_padding_mask=c.eq(PAD))
        h = self.norm(h)
        return self.h_cho(h), self.h_jung(h), self.h_jong(h)


def make_model(vocab, a, device):
    return DictTagger(vocab.size, d=a.d, layers=a.layers, heads=a.heads, ff=a.d * 4,
                      drop=a.dropout, max_pos=max(512, a.max_len)).to(device)


N_IN = 4 + N_FEAT  # 입력 텐서 수 (그 뒤는 타깃)


def collate(batch):
    n = max(len(b[0]) for b in batch)
    out = []
    for k in range(len(batch[0])):
        arr = np.full((len(batch), n), -100 if k >= N_IN else 0, np.int64)
        for i, b in enumerate(batch):
            arr[i, : len(b[k])] = b[k]
        out.append(torch.from_numpy(arr))
    return out


def build_variants(rows):
    """원문 어절 -> train 에서 관찰된 난독화 형태 목록 (글자 수가 같은 것만)."""
    v = {}
    for x, y in rows:
        xw, yw = x.split(" "), y.split(" ")
        if len(xw) != len(yw):
            continue
        for a, b in zip(xw, yw):
            if a and len(a) == len(b) and a != b:
                v.setdefault(b, set()).add(a)
    return {k: sorted(s) for k, s in v.items()}


def substitute_forms(x, y, variants, prob=0.3):
    """실제 입력 x 의 어절 일부를, 같은 원문 어절의 다른 관찰 형태로 교체."""
    xw, yw = x.split(" "), y.split(" ")
    if len(xw) != len(yw):
        return x
    for i, (a, b) in enumerate(zip(xw, yw)):
        if random.random() < prob:
            c = [w for w in variants.get(b, ()) if w != a and len(w) == len(a)]
            if c:
                xw[i] = random.choice(c)
    return " ".join(xw)


def partial_restore(x, y):
    """입력 어절의 일부(1개 ~ 1/3)를 정답 어절로 바꿔 둔다."""
    xw, yw = x.split(" "), y.split(" ")
    if len(xw) != len(yw) or len(xw) < 2:
        return x
    for i in random.sample(range(len(xw)), random.randint(1, max(1, len(xw) // 3))):
        if len(xw[i]) == len(yw[i]):
            xw[i] = yw[i]
    return " ".join(xw)


def smoothed_ce(logits, x_ids, mask, target, eps):
    """후보 마스크를 적용한 교차 엔트로피 + 라벨 스무딩(허용된 자모 안에서만 고르게 분산)."""
    allowed = mask[x_ids] | F.one_hot(target.clamp(min=0), logits.size(-1)).bool()
    lg = logits.float().masked_fill(~allowed, -1e4)
    if eps <= 0:
        return F.cross_entropy(lg.transpose(1, 2), target, ignore_index=-100)
    logp = F.log_softmax(lg, -1)
    valid = target != -100
    nll = -logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    smooth = -(logp * allowed).sum(-1) / allowed.sum(-1).clamp(min=1)
    return (((1 - eps) * nll + eps * smooth) * valid).sum() / valid.sum().clamp(min=1)


class PoolSet(Dataset):
    """epoch 마다 문장당: 실제 입력 1개 + 증강 pool 에서 무작위 aug 개 + 새 노이즈 fresh 개.

    v2 1차 실험(0.9606 < 02 의 0.9737) 진단 결과에 따른 대책:
      - fresh: 매번 새로 난독화한 샘플 (사전 특징 0 = 정보 없음). pool 4벌만 반복하면 과적합되어
               사전과 무관한 위치의 오류까지 3배로 늘었음 → 02 처럼 증강 다양성 확보.
      - feat_drop: 확률적으로 사전 특징을 통째로 끔. 학습 시 사전 1순위가 91.7% 맞다 보니 모델이 문맥 대신
                   사전을 베끼는 지름길을 배움(사전이 틀린 곳을 따라간 오류 2.2배) → 사전 없이도 판단하도록 강제.
    """

    def __init__(self, cache, vocab, aug, max_len, sampler=None, fresh=0, feat_drop=0.0,
                 gen=None, gen_ratio=0.0, variants=None, subst_ratio=0.0, partial=0.0, concat=0.0):
        self.items, self.feats, self.pool = cache["items"], cache["feats"], cache["pool"]
        self.n = len(self.items) // (1 + self.pool)
        self.vocab, self.aug, self.max_len = vocab, min(aug, self.pool), max_len
        self.sampler, self.fresh, self.feat_drop = sampler, fresh if sampler else 0, feat_drop
        # 11단계 (v3): '새 샘플' 자리(fresh)를 채우는 방법과 추가 변형
        #   gen      : 공개 언어모델이 생성한 리뷰(10단계)에 노이즈를 입힌 샘플 → train 에 없던 어휘 학습
        #   variants : 원문 어절 -> train 에서 실제 관찰된 난독화 형태들. 실제 입력의 어절 일부를 다른 관찰 형태로 교체
        #   partial  : 입력 어절 일부를 정답으로 미리 복원 (이미 깨끗한 단어는 그대로 두는 법 학습)
        #   concat   : 새 샘플을 이어 붙여 긴 입력을 만듦 (test 문장은 train 보다 훨씬 김)
        self.gen = gen or []
        self.gen_ratio = gen_ratio if self.gen else 0.0
        self.variants = variants or {}
        self.subst_ratio = subst_ratio if self.variants else 0.0
        self.partial, self.concat = partial, concat

    def _one(self, s):
        """사전 특징 없이 쓰는 새 샘플 한 개 (x, y)."""
        u = random.random()
        if u < self.gen_ratio:
            y = random.choice(self.gen)
            return self.sampler.corrupt(y), y
        x, y = self.items[s * (1 + self.pool)]
        if u < self.gen_ratio + self.subst_ratio:
            return substitute_forms(x, y, self.variants), y
        return self.sampler.corrupt(y), y

    def _fresh(self, s):
        x, y = self._one(s)
        if self.concat and random.random() < self.concat:
            target = random.randint(min(150, self.max_len), self.max_len)
            while len(x) < target:
                x2, y2 = self._one(random.randrange(self.n))
                x, y = x + " " + x2, y + " " + y2
        return x, y

    def __len__(self):
        return self.n * (1 + self.aug + self.fresh)

    def __getitem__(self, k):
        s, r = k % self.n, k // self.n
        if r > self.aug:  # 새 샘플 (사전 특징 없음)
            x, y = self._fresh(s)
            f = np.zeros((N_FEAT, len(x)), np.int16)
        else:
            v = 0 if r == 0 else random.randint(1, self.pool)
            idx = s * (1 + self.pool) + v
            (x, y), f = self.items[idx], self.feats[idx]
            if self.feat_drop and random.random() < self.feat_drop:
                f = np.zeros_like(f)
        if self.partial and random.random() < self.partial:
            x = partial_restore(x, y)
        if len(x) > self.max_len:
            st = random.randint(0, len(x) - self.max_len)
            x, y, f = x[st : st + self.max_len], y[st : st + self.max_len], f[:, st : st + self.max_len]
        c, a, b, g, ta, tb, tg = encode(self.vocab, x, y)
        return (c, a, b, g, *[f[j].astype(np.int64) for j in range(N_FEAT)], ta, tb, tg)


@torch.no_grad()
def predict_logprobs(model, vocab, masks, texts, feats, device, win=256, batch=64):
    """neural_tagger.predict_logprobs 와 같고 사전 특징만 추가."""
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
    acc = [[np.zeros((len(x), n), np.float32) for n in (19, 21, 28)] for x in texts]
    cnt = [np.zeros(len(x), np.float32) for x in texts]
    gm = [m.to(device) for m in masks]
    for i in range(0, len(jobs), batch):
        chunk = jobs[i : i + batch]
        items = [(*encode(vocab, texts[k][s:e]), *[feats[k][j, s:e].astype(np.int64) for j in range(N_FEAT)])
                 for k, s, e in chunk]
        t = [x.to(device) for x in collate(items)]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outs = model(*t)
        lps = [F.log_softmax(masked_logits(o, ids, m), -1).cpu().numpy()
               for o, ids, m in zip(outs, t[1:4], gm)]
        for j, (k, s, e) in enumerate(chunk):
            for h in range(3):
                acc[k][h][s:e] += lps[h][j, : e - s]
            cnt[k][s:e] += 1
    return [[h / np.maximum(cn, 1)[:, None] for h in a3] for a3, cn in zip(acc, cnt)]


def log_epoch(out_dir, row):
    path = os.path.join(out_dir, "tagger_v2_runs.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        with warnings.catch_warnings():  # 검증 안 한 epoch 의 빈 val_f1 때문에 나는 pandas FutureWarning 숨김
            warnings.simplefilter("ignore", FutureWarning)
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


HP_KEYS = ["run_name", "full", "pool", "aug", "fresh", "feat_drop", "gen_ratio", "subst_ratio", "partial", "concat",
           "label_smoothing", "ema", "dict_min_len", "dict_oov", "max_len", "cand_thr", "d",
           "layers", "heads", "dropout", "epochs", "batch", "lr", "seed", "init_ckpt"]


def train(args, device):
    path = cache_path(args)
    if not os.path.exists(path):
        raise SystemExit(f"{path} 가 없습니다. 먼저 --mode prep {'--full ' if args.full else ''}을 실행하세요.")
    with open(path, "rb") as fp:
        cache = pickle.load(fp)
    ca = cache["args"]
    for k in ("dict_min_len", "dict_oov", "cand_thr", "pool"):  # 캐시를 만든 설정을 따름
        setattr(args, k, ca[k])
    print(f"  cache {path} (pool {ca['pool']}, dict_oov {ca['dict_oov']}, cand_thr {ca['cand_thr']})")

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
    gen = []
    if args.gen_file:
        with open(args.gen_file, encoding="utf-8") as fp:
            gen = [ln.strip() for ln in fp if ln.strip()]
    variants = build_variants(nc.aligned_rows(tr)) if args.subst_ratio > 0 else None
    ds = PoolSet(cache, vocab, args.aug, args.max_len, sampler=sampler, fresh=args.fresh, feat_drop=args.feat_drop,
                 gen=gen, gen_ratio=args.gen_ratio, variants=variants, subst_ratio=args.subst_ratio,
                 partial=args.partial, concat=args.concat)
    if gen or variants or args.partial or args.concat or args.label_smoothing or args.ema:
        print(f"  v3 옵션: 생성 문장 {len(gen)}개 (새 샘플의 {ds.gen_ratio:.0%}) | 관찰 형태 치환 {ds.subst_ratio:.0%} "
              f"(어절 {len(variants or {})}종) | 부분 복원 {args.partial:.0%} | 이어 붙이기 {args.concat:.0%} "
              f"| 라벨 스무딩 {args.label_smoothing} | EMA {args.ema}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=args.loader_workers,
                    drop_last=True, persistent_workers=args.loader_workers > 0)
    print(f"  params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | 문장 {ds.n} x "
          f"(실제 1 + pool {ds.aug} + 새 노이즈 {ds.fresh}) | 사전 특징 끄기 {ds.feat_drop:.0%} | steps/epoch {len(dl)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(dl)
    warm = int(0.05 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(1, s / total)))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema, step = None, 0  # 가중치 지수이동평균: 평가·저장은 EMA 가중치로 (초반에는 감쇠를 작게)
    if args.ema > 0:
        ema = copy.deepcopy(model).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
    sfx = f"_{args.run_name}" if args.run_name else ""  # 실행마다 다른 이름으로 저장 (덮어쓰기 방지)
    ckpt = os.path.join(args.out_dir, f"tagger_v2_full{sfx}.pt" if args.full else f"tagger_v2_best{sfx}.pt")
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
            loss = sum(smoothed_ce(o, ids, m, t, args.label_smoothing)
                       for o, ids, m, t in zip(outs, inp[1:4], gmasks, tgt))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            step += 1
            if ema is not None:
                d = min(args.ema, (1 + step) / (10 + step))
                with torch.no_grad():
                    for q, w_ in zip(ema.parameters(), model.parameters()):
                        q.mul_(d).add_(w_.detach(), alpha=1 - d)
        sec = time.time() - t0
        msg = f"epoch {ep:3d} | loss {tot / len(dl):.4f} | {sec:.0f}s"
        eval_model = ema if ema is not None else model
        state = {"model": eval_model.state_dict(), "args": vars(args), "epoch": ep}
        f1 = None
        if va is not None and (ep % args.eval_every == 0 or ep == args.epochs):
            lps = predict_logprobs(eval_model, vocab, masks, list(va.input), cache["val_feats"], device,
                                   win=args.max_len)
            preds = [dec_.decode(x, lp) for x, lp in zip(va.input, lps)]
            f1 = nc.mean_f1(preds, list(va.output))
            msg += f" | val F1 {f1:.4f}"
            if f1 > best:
                best = f1
                torch.save(state, ckpt)
                va.assign(pred=preds, f1=[nc.f1_score(p, g) for p, g in zip(preds, va.output)]).to_csv(
                    os.path.join(args.out_dir, f"val_pred_tagger_v2{sfx}.csv"), index=False, encoding="utf-8-sig")
                msg += " *"
        elif va is None:
            torch.save(state, ckpt)
        print(msg, flush=True)
        log_epoch(args.out_dir, {"run_id": run_id, "epoch": ep, "loss": round(tot / len(dl), 5),
                                 "val_f1": None if f1 is None else round(f1, 5),
                                 "n_val": 0 if va is None else len(va), "sec": round(sec, 1), **hp})
    if va is not None:
        print(f"best val F1 {best:.4f} -> {ckpt}  (02 태거 단독 0.9737 / 01 노이즈 채널 0.9834 와 비교)")


def predict(args, device):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    sfx = f"_{args.run_name}" if args.run_name else ""
    ckpt = args.ckpt or os.path.join(args.out_dir, f"tagger_v2_best{sfx}.pt")
    st = torch.load(ckpt, map_location=device)
    a = argparse.Namespace(**st["args"])
    print(f"loaded {ckpt} (epoch {st['epoch']})")
    ch = nc.JamoChannel(nc.aligned_rows(full), cand_thr=a.cand_thr)
    wd, vocab_chars = build_dict(full, ch, a)  # test 는 train 전체로 만든 사전 (자기 문장 제외 불필요)
    t = time.time()
    feats = compute_feats(wd, list(test.input), [None] * len(test), args.workers)  # 프로세스 분기는 GPU 사용 전에
    print(f"  test 특징 {time.time() - t:.0f}s")
    vocab = CharVocab(list(full.input) + list(full.output) + list(test.input))
    model = make_model(vocab, a, device)
    model.load_state_dict(st["model"])
    lps = predict_logprobs(model, vocab, build_masks(ch), list(test.input), feats, device, win=a.max_len)
    dec_ = CandidateDecoder(ch, vocab_chars)
    preds = [dec_.decode(x, lp) for x, lp in zip(test.input, lps)]
    sub["output"] = sub.ID.map(dict(zip(test.ID, preds)))
    out = os.path.join(args.out_dir, f"submission_tagger_v2{sfx}.csv")
    sub.to_csv(out, index=False, encoding="utf-8-sig")
    with open(os.path.join(args.out_dir, f"submission_tagger_v2{sfx}_params.json"), "w", encoding="utf-8") as fp:
        json.dump({"ckpt": ckpt, "epoch": st["epoch"], "train_args": st["args"],
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, ensure_ascii=False, indent=1)
    print(f"saved {out} (+ _params.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prep", "train", "predict"], default="train")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--ckpt", default="", help="predict 때 체크포인트 (기본 outputs/tagger_v2_best.pt)")
    ap.add_argument("--full", action="store_true", help="val 없이 전체 train 으로 (prep·train 모두 같이 지정)")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    # 사전 특징 (prep 에서 정하고 train 은 캐시 설정을 따름)
    ap.add_argument("--pool", type=int, default=4, help="문장당 미리 만들어 둘 증강 변형 수")
    ap.add_argument("--dict_min_len", type=int, default=2)
    ap.add_argument("--dict_oov", type=float, default=0.5, help="01 sweep 최고값")
    ap.add_argument("--cand_thr", type=float, default=0.002)
    # 학습
    ap.add_argument("--aug", type=int, default=1, help="epoch 마다 문장당 pool 에서 뽑을 증강 수 (<= pool, 사전 특징 있음)")
    ap.add_argument("--fresh", type=int, default=2, help="epoch 마다 문장당 새로 만드는 노이즈 샘플 수 (사전 특징 없음)")
    ap.add_argument("--feat_drop", type=float, default=0.3, help="사전 특징을 통째로 끌 확률 (사전 베끼기 방지)")
    # 11단계 (v3): 기본값은 모두 꺼짐 = 기존 v2 와 동일
    ap.add_argument("--gen_file", default="", help="공개 언어모델이 생성한 리뷰 파일 (10단계). 노이즈를 입혀 학습 샘플로 사용")
    ap.add_argument("--gen_ratio", type=float, default=0.0, help="새 샘플 중 생성 리뷰로 만드는 비율")
    ap.add_argument("--subst_ratio", type=float, default=0.0, help="새 샘플 중 '실제 관찰된 난독화 형태로 치환'으로 만드는 비율")
    ap.add_argument("--partial", type=float, default=0.0, help="입력 어절 일부를 정답으로 미리 복원할 확률")
    ap.add_argument("--concat", type=float, default=0.0, help="새 샘플을 이어 붙여 긴 입력으로 만들 확률")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0, help="가중치 EMA 감쇠율 (예: 0.999, 0 = 끔)")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--run_name", default="",
                    help="결과 파일 이름 꼬리표. 예: scratch → tagger_v2_best_scratch.pt, val_pred_tagger_v2_scratch.csv,"
                         " submission_tagger_v2_scratch.csv (predict 도 같은 값으로)")
    ap.add_argument("--init_ckpt", default="", help="02 체크포인트에서 이어 학습 (예: outputs/tagger_best.pt)")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1),
                    help="사전 특징 계산 프로세스 수 (너무 많으면 Colab 메모리 압박으로 드라이브 연결이 끊길 수 있음)")
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
