"""
난독화 한글 리뷰 복원 - 2단계: 글자 단위 Transformer 태거 (GPU)

입력과 정답 길이가 1:1 이므로 seq2seq 대신 "글자별 분류"로 푼다.
- 입력 임베딩 = 글자 + 초성 + 중성 + 종성 + 위치  (드문 난독 글자도 자모로 통계를 공유)
- 출력 = 한글 위치마다 원본 초성(19) / 중성(21) / 종성(28) 3개 헤드
- 후보 마스크 = 노이즈 채널의 역방향 후보(그 난독 자모를 만들 수 있는 원본 자모)만 허용
- 노이즈 증강 = 채널 확률표(받침·연음 결합 포함)로 train output 을 매 epoch 새로 난독화
- 디코딩 = 후보 음절(train output 에 나온 글자) 중 세 헤드 log-prob 합이 최대인 것

사용 예 (Colab, GPU 런타임):
    !python neural_tagger.py --mode train            # train 90% 학습, val 10% 로 best 저장
    !python neural_tagger.py --mode predict          # best 체크포인트로 test 예측
    !python neural_tagger.py --mode train --full --epochs 30   # val 포함 전체로 학습(고정 epoch)
"""
import argparse
import json
import math
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from noisy_channel import (
    IEUNG, N_CHO, N_JONG, N_JUNG, JamoChannel, aligned_rows, comp, dec, f1_score, is_hangul, mean_f1,
)

PAD, UNK = 0, 1
N_HANGUL = 11172


# --------------------------------------------------------------------------- #
# 입력 인코딩
# --------------------------------------------------------------------------- #
class CharVocab:
    """PAD, UNK, 한글 음절 11172개 전부, 그 외 문자(train/test 에서 수집)."""

    def __init__(self, texts):
        others = sorted({c for t in texts for c in t if not is_hangul(c)})
        self.other = {c: i + 2 + N_HANGUL for i, c in enumerate(others)}
        self.size = 2 + N_HANGUL + len(others)

    def id(self, c):
        if is_hangul(c):
            return 2 + ord(c) - 0xAC00
        return self.other.get(c, UNK)


def encode(vocab, x, y=None):
    """x -> (글자 id, 초성+1, 중성+1, 종성+1) ; 한글이 아니면 자모 id 0. y 가 있으면 자모 타깃(-100=무시)."""
    n = len(x)
    c = np.zeros(n, np.int64)
    a = np.zeros(n, np.int64)
    b = np.zeros(n, np.int64)
    g = np.zeros(n, np.int64)
    for i, ch in enumerate(x):
        c[i] = vocab.id(ch)
        if is_hangul(ch):
            j = dec(ch)
            a[i], b[i], g[i] = j[0] + 1, j[1] + 1, j[2] + 1
    if y is None:
        return c, a, b, g
    ta = np.full(n, -100, np.int64)
    tb = np.full(n, -100, np.int64)
    tg = np.full(n, -100, np.int64)
    for i, (xc, yc) in enumerate(zip(x, y)):
        if is_hangul(xc) and is_hangul(yc):
            ta[i], tb[i], tg[i] = dec(yc)
    return c, a, b, g, ta, tb, tg


# --------------------------------------------------------------------------- #
# 노이즈 증강: 채널 확률표로 원본 -> 난독 샘플링
# --------------------------------------------------------------------------- #
class NoiseSampler:
    def __init__(self, ch):
        cw = lambda row: np.cumsum(np.exp(np.asarray(row, np.float64)))
        self.cho = [cw(r) for r in ch.lp_cho]
        self.cho_ieung0 = cw(ch.lp_cho_ieung[0])   # 구간 첫 글자가 ㅇ 초성
        self.jung = [cw(r) for r in ch.lp_jung]
        self.jong_plain = [cw(r) for r in ch.lp_jong_plain]
        self.pair = [cw(r) for r in ch.lp_pair]      # 인덱스 = x받침*N_CHO + x초성

    @staticmethod
    def _draw(cum):
        return int(np.searchsorted(cum, random.random() * cum[-1], side="right"))

    def corrupt(self, y):
        xj = [None] * len(y)  # 난독 자모 [초, 중, 종]
        prev = None           # 직전 한글 위치
        for i, c in enumerate(y):
            if not is_hangul(c):
                if prev is not None:
                    xj[prev][2] = self._draw(self.jong_plain[dec(y[prev])[2]])
                prev = None
                continue
            yc, yv, _ = dec(c)
            xv = self._draw(self.jung[yv])
            if prev is not None:
                pJ = dec(y[prev])[2]
                if yc == IEUNG:  # 앞 받침과 현재 초성을 함께 샘플링(연음)
                    pg, xc = divmod(self._draw(self.pair[pJ]), N_CHO)
                else:
                    pg, xc = self._draw(self.jong_plain[pJ]), self._draw(self.cho[yc])
                xj[prev][2] = pg
            else:
                xc = self._draw(self.cho_ieung0) if yc == IEUNG else self._draw(self.cho[yc])
            xj[i] = [xc, xv, 0]
            prev = i
        if prev is not None:
            xj[prev][2] = self._draw(self.jong_plain[dec(y[prev])[2]])
        return "".join(comp(*j) if j else c for j, c in zip(xj, y))


# --------------------------------------------------------------------------- #
# 데이터셋
# --------------------------------------------------------------------------- #
class TrainSet(Dataset):
    """실제 쌍 + 증강 쌍(aug 배). 길이가 max_len 보다 길면 무작위 구간을 자름."""

    def __init__(self, rows, sampler, vocab, aug, max_len):
        self.rows, self.sampler, self.vocab = rows, sampler, vocab
        self.aug, self.max_len = aug, max_len

    def __len__(self):
        return len(self.rows) * (1 + self.aug)

    def __getitem__(self, k):
        x, y = self.rows[k % len(self.rows)]
        if k >= len(self.rows):  # 증강: 정답 문장을 새로 난독화
            x = self.sampler.corrupt(y)
        if len(x) > self.max_len:
            s = random.randint(0, len(x) - self.max_len)
            x, y = x[s : s + self.max_len], y[s : s + self.max_len]
        return encode(self.vocab, x, y)


def collate(batch):
    n = max(len(b[0]) for b in batch)
    out = []
    for k in range(len(batch[0])):
        fill = -100 if k >= 4 else 0
        arr = np.full((len(batch), n), fill, np.int64)
        for i, b in enumerate(batch):
            arr[i, : len(b[k])] = b[k]
        out.append(torch.from_numpy(arr))
    return out


# --------------------------------------------------------------------------- #
# 모델
# --------------------------------------------------------------------------- #
class Tagger(nn.Module):
    def __init__(self, n_char, d=384, layers=6, heads=6, ff=1536, drop=0.1, max_pos=512):
        super().__init__()
        self.char = nn.Embedding(n_char, d, padding_idx=PAD)
        self.cho = nn.Embedding(N_CHO + 1, d)
        self.jung = nn.Embedding(N_JUNG + 1, d)
        self.jong = nn.Embedding(N_JONG + 1, d)
        self.pos = nn.Embedding(max_pos, d)
        self.drop = nn.Dropout(drop)
        layer = nn.TransformerEncoderLayer(
            d, heads, ff, drop, activation="gelu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.h_cho = nn.Linear(d, N_CHO)
        self.h_jung = nn.Linear(d, N_JUNG)
        self.h_jong = nn.Linear(d, N_JONG)

    def forward(self, c, a, b, g):
        pos = torch.arange(c.size(1), device=c.device)
        h = self.char(c) + self.cho(a) + self.jung(b) + self.jong(g) + self.pos(pos)[None]
        h = self.enc(self.drop(h), src_key_padding_mask=c.eq(PAD))
        h = self.norm(h)
        return self.h_cho(h), self.h_jung(h), self.h_jong(h)


def build_masks(ch):
    """난독 자모 id(+1, 0=비한글) -> 허용 원본 자모 bool. 비한글 행은 전부 허용(손실에서 무시됨)."""
    def one(inv, n):
        m = torch.zeros(n + 1, n, dtype=torch.bool)
        m[0] = True
        for t in range(n):
            m[t + 1, inv[t]] = True
        return m
    return one(ch.inv_cho, N_CHO), one(ch.inv_jung, N_JUNG), one(ch.inv_jong, N_JONG)


def masked_logits(logits, x_ids, mask, target=None):
    allowed = mask[x_ids]  # (B, L, C)
    if target is not None:  # 정답은 항상 허용 (마스크 밖 정답에서 손실이 무한대가 되지 않게)
        allowed = allowed | F.one_hot(target.clamp(min=0), logits.size(-1)).bool()
    return logits.float().masked_fill(~allowed, -1e4)


# --------------------------------------------------------------------------- #
# 추론
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_logprobs(model, vocab, masks, texts, device, win=256, batch=64):
    """긴 문장은 겹치는 창(stride=win/2)으로 나눠 예측 후 위치별 log-prob 평균."""
    model.eval()
    jobs = []  # (문장 번호, 시작 위치, 조각)
    for k, x in enumerate(texts):
        if len(x) <= win:
            jobs.append((k, 0, x))
        else:
            starts = list(range(0, len(x) - win + 1, win // 2))
            if starts[-1] != len(x) - win:
                starts.append(len(x) - win)
            jobs += [(k, s, x[s : s + win]) for s in starts]
    acc = [[np.zeros((len(x), n), np.float32) for n in (N_CHO, N_JUNG, N_JONG)] for x in texts]
    cnt = [np.zeros(len(x), np.float32) for x in texts]
    for i in range(0, len(jobs), batch):
        chunk = jobs[i : i + batch]
        c, a, b, g = [t.to(device) for t in collate([encode(vocab, s) for _, _, s in chunk])]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outs = model(c, a, b, g)
        lps = [
            F.log_softmax(masked_logits(o, ids, m.to(device)), -1).cpu().numpy()
            for o, ids, m in zip(outs, (a, b, g), masks)
        ]
        for j, (k, s, sub) in enumerate(chunk):
            L = len(sub)
            for h in range(3):
                acc[k][h][s : s + L] += lps[h][j, :L]
            cnt[k][s : s + L] += 1
    return [[h / np.maximum(cn, 1)[:, None] for h in a3] for a3, cn in zip(acc, cnt)]


class CandidateDecoder:
    """위치별 후보 음절(train output 에 나온 글자) 중 세 헤드 log-prob 합 최대. 후보가 없으면 헤드별 argmax."""

    def __init__(self, ch, vocab_chars):
        self.ch, self.vocab_chars = ch, vocab_chars
        self.cache = {}

    def cands(self, xc):
        v = self.cache.get(xc)
        if v is None:
            v = [(y, *dec(y)) for y, *_ in self.ch.candidates(xc, self.vocab_chars)]
            self.cache[xc] = v
        return v

    def decode(self, x, lp):
        la, lb, lg = lp
        out = list(x)
        for i, xc in enumerate(x):
            if not is_hangul(xc):
                continue
            best, bs = None, -1e18
            for y, a, b, g in self.cands(xc):
                s = la[i, a] + lb[i, b] + lg[i, g]
                if s > bs:
                    best, bs = y, s
            out[i] = best or comp(int(la[i].argmax()), int(lb[i].argmax()), int(lg[i].argmax()))
        return "".join(out)


# --------------------------------------------------------------------------- #
# 학습
# --------------------------------------------------------------------------- #
def split_train(train, val_ratio, seed):
    """noisy_channel.py 와 같은 분할 (결과 비교 가능)."""
    idx = list(range(len(train)))
    random.Random(seed).shuffle(idx)
    n_val = int(len(idx) * val_ratio)
    return train.iloc[idx[n_val:]].reset_index(drop=True), train.iloc[idx[:n_val]].reset_index(drop=True)


def evaluate(model, vocab, masks, dec_, va, device, args):
    lps = predict_logprobs(model, vocab, masks, list(va.input), device, win=args.max_len)
    preds = [dec_.decode(x, lp) for x, lp in zip(va.input, lps)]
    return mean_f1(preds, list(va.output)), preds


HP_KEYS = ["full", "aug", "max_len", "cand_thr", "d", "layers", "heads", "dropout",
           "epochs", "batch", "lr", "seed"]


def log_epoch(out_dir, row):
    """epoch 별 손실·검증 F1·하이퍼파라미터를 outputs/tagger_runs.csv 에 누적 기록."""
    path = os.path.join(out_dir, "tagger_runs.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        with warnings.catch_warnings():  # 검증 안 한 epoch 의 빈 val_f1 때문에 나는 pandas FutureWarning 숨김
            warnings.simplefilter("ignore", FutureWarning)
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def train(args, device):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    tr, va = (full, None) if args.full else split_train(full, args.val_ratio, args.seed)
    if va is not None and args.val_limit:
        va = va.iloc[: args.val_limit]

    rows = aligned_rows(tr)
    ch = JamoChannel(rows, cand_thr=args.cand_thr)
    vocab = CharVocab(list(full.input) + list(full.output) + list(test.input))
    masks = build_masks(ch)
    dec_ = CandidateDecoder(ch, {c for t in tr.output for c in t if is_hangul(c)})
    sampler = NoiseSampler(ch)

    ds = TrainSet(rows, sampler, vocab, args.aug, args.max_len)
    dl = DataLoader(
        ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
        num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0,
    )
    model = Tagger(vocab.size, d=args.d, layers=args.layers, heads=args.heads, ff=args.d * 4,
                   drop=args.dropout, max_pos=max(512, args.max_len)).to(device)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | "
          f"train rows {len(rows)} x (1+{args.aug}) | steps/epoch {len(dl)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(dl)
    warm = int(0.05 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(1, s / total)))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    gmasks = [m.to(device) for m in masks]

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, "tagger_full.pt" if args.full else "tagger_best.pt")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    hp = {k: getattr(args, k) for k in HP_KEYS}
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot_loss = time.time(), 0.0
        for c, a, b, g, ta, tb, tg in dl:
            c, a, b, g, ta, tb, tg = [t.to(device, non_blocking=True) for t in (c, a, b, g, ta, tb, tg)]
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                outs = model(c, a, b, g)
            loss = sum(
                F.cross_entropy(masked_logits(o, ids, m, t).transpose(1, 2), t, ignore_index=-100)
                for o, ids, m, t in zip(outs, (a, b, g), gmasks, (ta, tb, tg))
            )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item()
        sec = time.time() - t0
        msg = f"epoch {ep:3d} | loss {tot_loss / len(dl):.4f} | {sec:.0f}s"
        state = {"model": model.state_dict(), "args": vars(args), "epoch": ep}
        f1 = None
        if va is not None and (ep % args.eval_every == 0 or ep == args.epochs):
            f1, preds = evaluate(model, vocab, masks, dec_, va, device, args)
            msg += f" | val F1 {f1:.4f}"
            if f1 > best:
                best = f1
                torch.save(state, ckpt)
                va.assign(pred=preds, f1=[f1_score(p, g) for p, g in zip(preds, va.output)]).to_csv(
                    os.path.join(args.out_dir, "val_pred_tagger.csv"), index=False, encoding="utf-8-sig"
                )
                msg += " *"
        elif va is None:
            torch.save(state, ckpt)
        print(msg, flush=True)
        log_epoch(args.out_dir, {"run_id": run_id, "epoch": ep, "loss": round(tot_loss / len(dl), 5),
                                 "val_f1": None if f1 is None else round(f1, 5),
                                 "n_val": 0 if va is None else len(va), "sec": round(sec, 1), **hp})
    if va is not None:
        print(f"best val F1 {best:.4f} -> {ckpt}  (기록: outputs/tagger_runs.csv, run_id {run_id})")


def predict(args, device):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    ckpt_path = args.ckpt or os.path.join(args.out_dir, "tagger_best.pt")
    state = torch.load(ckpt_path, map_location=device)
    a = argparse.Namespace(**state["args"])
    print(f"loaded {ckpt_path} (epoch {state['epoch']})")

    ch = JamoChannel(aligned_rows(full), cand_thr=a.cand_thr)
    vocab = CharVocab(list(full.input) + list(full.output) + list(test.input))
    model = Tagger(vocab.size, d=a.d, layers=a.layers, heads=a.heads, ff=a.d * 4,
                   drop=a.dropout, max_pos=max(512, a.max_len)).to(device)
    model.load_state_dict(state["model"])
    # 마스크는 체크포인트 학습 때와 같은 기준으로 다시 만들되, 후보 음절은 전체 train output 사용
    masks = build_masks(ch)
    dec_ = CandidateDecoder(ch, {c for t in full.output for c in t if is_hangul(c)})
    lps = predict_logprobs(model, vocab, masks, list(test.input), device, win=a.max_len)
    preds = [dec_.decode(x, lp) for x, lp in zip(test.input, lps)]
    sub["output"] = sub.ID.map(dict(zip(test.ID, preds)))
    path = os.path.join(args.out_dir, "submission_tagger.csv")
    sub.to_csv(path, index=False, encoding="utf-8-sig")
    with open(os.path.join(args.out_dir, "submission_tagger_params.json"), "w", encoding="utf-8") as f:
        json.dump({"ckpt": ckpt_path, "epoch": state["epoch"], "train_args": state["args"],
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=1)
    print(f"saved {path} (+ _params.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "predict"], default="train")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--ckpt", default="", help="predict 때 사용할 체크포인트 (기본: outputs/tagger_best.pt)")
    ap.add_argument("--full", action="store_true", help="val 없이 전체 train 으로 학습")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    # 데이터
    ap.add_argument("--aug", type=int, default=2, help="실제 쌍 1개당 증강 쌍 수 (매 epoch 새로 생성)")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--cand_thr", type=float, default=0.002, help="후보 마스크 기준 (작을수록 넓게 허용)")
    # 모델
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    # 학습
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(vars(args), "| device", device)
    if device.type != "cuda":
        print("경고: GPU 가 없습니다. Colab 런타임 유형을 GPU 로 바꾸세요.")
    (train if args.mode == "train" else predict)(args, device)


if __name__ == "__main__":
    main()
