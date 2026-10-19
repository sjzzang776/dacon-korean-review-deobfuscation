"""
난독화 한글 리뷰 복원 - 12b: LLM 없이 쓰는 글자 단위 언어모델 (처음부터 학습)

12 의 빈칸 채우기 모델은 가린 글자에서만 학습 신호가 나와 60 epoch 뒤에도 정확도 14% 에 그쳤다
(파이프라인 검증 0.9583, 01 의 0.9834 보다 낮음). 12b 는 '다음 글자 예측' 으로 바꾼다.

  - 모든 글자 위치에서 학습 신호가 나온다 (12 의 6~7배).
  - 한 모델이 정방향·역방향을 모두 배운다 (첫 토큰으로 방향 표시). 역방향은 문장을 뒤집어 학습.
  - 출력은 train 에 나온 글자 전체에 대한 softmax (12 의 초·중·종성 독립 가정 제거), 입력·출력 임베딩 공유.
  - train 정답 리뷰만 사용. 사전학습 모델·외부 데이터·LLM 생성 문장 없음.

채점: CharLMScorer(texts) -> 문자열마다 (정방향 log P + 역방향 log P) / 2.  anchor_lm12.py 에서 사용.

사용 예 (Colab, GPU):
    !python -u char_lm.py                       # train 90% 로 학습, 검증 손실 최저 시점 저장 -> outputs/charlm_best.pt
    !python -u char_lm.py --full --epochs 18    # test 용: train 전체, 고정 epoch -> outputs/charlm_full.pt
"""
import argparse
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import noisy_channel as nc
from neural_tagger import split_train

PAD, UNK, BOS_F, BOS_B, EOS = 0, 1, 2, 3, 4
N_SPECIAL = 5


class CharLM(nn.Module):
    def __init__(self, n_vocab, d=384, layers=6, heads=6, drop=0.2, max_pos=320):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_pos, d)
        self.drop = nn.Dropout(drop)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.bias = nn.Parameter(torch.zeros(n_vocab))
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, ids):
        n = ids.size(1)
        h = self.emb(ids) * math.sqrt(self.emb.embedding_dim) + self.pos(torch.arange(n, device=ids.device))[None]
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=ids.device), 1)
        h = self.enc(self.drop(h), mask=causal, src_key_padding_mask=ids.eq(PAD))
        return F.linear(self.norm(h), self.emb.weight, self.bias)  # 입력·출력 임베딩 공유


def make_vocab(texts):
    return sorted({c for t in texts for c in t})


def to_ids(c2i, text, backward):
    body = [c2i.get(c, UNK) for c in (reversed(text) if backward else text)]
    return [BOS_B if backward else BOS_F] + body


class LMSet(Dataset):
    """문장마다 정방향·역방향 한 번씩. crop 확률로 문장 중간 조각을 쓴다 (채점 때는 문맥 창이 문장 중간에서 시작하므로)."""

    def __init__(self, texts, c2i, max_len, crop=0.5, unk_drop=0.01, train=True):
        self.texts, self.c2i, self.max_len, self.crop, self.unk_drop, self.train = texts, c2i, max_len, crop, unk_drop, train

    def __len__(self):
        return 2 * len(self.texts)

    def __getitem__(self, k):
        t, backward = self.texts[k // 2], k % 2 == 1
        whole = True
        if self.train and len(t) > 30 and random.random() < self.crop:
            n = random.randint(20, min(len(t), self.max_len))
            s = random.randint(0, len(t) - n)
            whole = s + n == len(t) if not backward else s == 0  # 읽는 방향의 끝이 진짜 문장 끝일 때만 EOS
            t = t[s : s + n]
        elif len(t) > self.max_len:
            s = random.randint(0, len(t) - self.max_len) if self.train else 0
            t, whole = t[s : s + self.max_len], False
        ids = to_ids(self.c2i, t, backward) + ([EOS] if whole else [])
        x, y = np.array(ids[:-1], np.int64), np.array(ids[1:], np.int64)
        if self.train and self.unk_drop:  # 입력 글자 일부를 UNK 로 (모르는 글자가 문맥에 있을 때 대비)
            m = np.random.rand(len(x)) < self.unk_drop
            m[0] = False
            x = np.where(m, UNK, x)
        return x, y


def collate(batch):
    n = max(len(x) for x, _ in batch)
    xs = np.zeros((len(batch), n), np.int64)
    ys = np.full((len(batch), n), -100, np.int64)
    for i, (x, y) in enumerate(batch):
        xs[i, : len(x)], ys[i, : len(y)] = x, y
    return torch.from_numpy(xs), torch.from_numpy(ys)


@torch.no_grad()
def evaluate(model, dl, device):
    """글자당 손실 (정방향, 역방향)."""
    model.eval()
    tot, cnt = [0.0, 0.0], [0, 0]
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
        loss = F.cross_entropy(out.float().transpose(1, 2), y, ignore_index=-100, reduction="none").sum(1)
        n = (y >= 0).sum(1)
        for d, b in ((0, x[:, 0] == BOS_F), (1, x[:, 0] == BOS_B)):
            tot[d] += float(loss[b].sum())
            cnt[d] += int(n[b].sum())
    return tot[0] / max(cnt[0], 1), tot[1] / max(cnt[1], 1)


def kn_loss(train_texts, val_texts, order=5):
    """비교 기준: 01 이 쓰는 문자 5-gram 의 글자당 손실 (정방향, 문장 끝 포함)."""
    lm = nc.KNLM(train_texts, order=order)
    tot = n = 0
    for t in val_texts:
        s = nc.BOS * (order - 1) + t + nc.EOS
        for i in range(order - 1, len(s)):
            tot -= lm.logp(s[i - order + 1 : i], s[i])
            n += 1
    return tot / max(n, 1)


def train(args, device):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    tr, va = (full, None) if args.full else split_train(full, args.val_ratio, args.seed)
    clean = lambda s: [str(y) for y in s if str(y).strip()]
    texts = clean(tr.output)
    chars = make_vocab(texts)
    c2i = {c: i + N_SPECIAL for i, c in enumerate(chars)}
    hp = {k: getattr(args, k) for k in ("d", "layers", "heads", "dropout", "max_len")}
    model = CharLM(len(chars) + N_SPECIAL, args.d, args.layers, args.heads, args.dropout, args.max_len + 8).to(device)
    dl = DataLoader(LMSet(texts, c2i, args.max_len, args.crop, args.unk_drop), batch_size=args.batch, shuffle=True,
                    collate_fn=collate, num_workers=args.loader_workers, drop_last=True,
                    persistent_workers=args.loader_workers > 0)
    vdl = None
    if va is not None:
        vt = clean(va.output)
        vdl = DataLoader(LMSet(vt, c2i, args.max_len, train=False), batch_size=256, collate_fn=collate)
        print(f"  비교 기준: 01 의 문자 5-gram 검증 손실(글자당) {kn_loss(texts, vt):.4f}")
    print(f"  params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | 글자 종류 {len(chars)} | "
          f"학습 문장 {len(texts)} (x2 방향) | steps/epoch {len(dl)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / max(1, int(0.03 * total))) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, "charlm_full.pt" if args.full else "charlm_best.pt")
    best, rows = 1e9, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x)
            loss = F.cross_entropy(out.float().transpose(1, 2), y, ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        msg = f"epoch {ep:3d} | 학습 손실 {tot / len(dl):.4f} | {time.time() - t0:.0f}s"
        state = {"model": model.state_dict(), "hp": hp, "chars": chars, "epoch": ep, "full": args.full}
        vf = vb = None
        if vdl is not None:
            vf, vb = evaluate(model, vdl, device)
            msg += f" | 검증 손실(글자당): 정방향 {vf:.4f}, 역방향 {vb:.4f}"
            if (vf + vb) / 2 < best:
                best = (vf + vb) / 2
                torch.save(state, ckpt)
                msg += " *"
        else:
            torch.save(state, ckpt)
        print(msg, flush=True)
        rows.append({"epoch": ep, "train_loss": round(tot / len(dl), 5), "val_fwd": vf, "val_bwd": vb, **hp,
                     "lr": args.lr, "weight_decay": args.weight_decay, "crop": args.crop, "epochs": args.epochs,
                     "full": args.full})
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"charlm_runs_{'full' if args.full else 'split'}.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"저장: {ckpt}" + (f" (검증 손실 최저 {best:.4f})" if vdl is not None else ""))


class CharLMScorer:
    """texts -> 문자열마다 (정방향 log P + 역방향 log P) / 2.  같은 문자열은 캐시."""

    def __init__(self, ckpt, batch_tokens=131072):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st = torch.load(ckpt, map_location=self.device)
        hp = st["hp"]
        self.c2i = {c: i + N_SPECIAL for i, c in enumerate(st["chars"])}
        self.max_len = hp["max_len"]
        self.model = CharLM(len(st["chars"]) + N_SPECIAL, hp["d"], hp["layers"], hp["heads"], hp["dropout"],
                            hp["max_len"] + 8).to(self.device)
        self.model.load_state_dict(st["model"])
        self.model.eval()
        self.batch_tokens, self.cache, self.n_forward = batch_tokens, {}, 0
        print(f"  글자 언어모델 {ckpt} (epoch {st['epoch']}, "
              f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M) on {self.device}")

    @torch.no_grad()
    def __call__(self, texts):
        todo = sorted({t for t in texts if t not in self.cache}, key=len)
        i = 0
        while i < len(todo):
            L = min(len(todo[min(i + 511, len(todo) - 1)]), self.max_len) + 1
            chunk = todo[i : i + max(1, min(512, self.batch_tokens // (2 * L)))]
            seqs = [to_ids(self.c2i, t[: self.max_len], b) for t in chunk for b in (False, True)]
            n = max(len(s) for s in seqs)
            ids = torch.zeros((len(seqs), n), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, : len(s)] = torch.tensor(s)
            ids = ids.to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                out = self.model(ids[:, :-1])
            lp = F.log_softmax(out.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            lp = (lp * ids[:, 1:].ne(PAD)).sum(1).view(len(chunk), 2).mean(1).cpu().tolist()
            for t, v in zip(chunk, lp):
                self.cache[t] = v
            self.n_forward += len(seqs)
            i += len(chunk)
        return [self.cache[t] for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--full", action="store_true", help="train 전체로 학습 (test 예측용, 고정 epoch)")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--crop", type=float, default=0.5, help="문장 중간 조각으로 학습할 확률")
    ap.add_argument("--unk_drop", type=float, default=0.01)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--loader_workers", type=int, default=2)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(vars(args), "| device", device)
    train(args, device)


if __name__ == "__main__":
    main()
