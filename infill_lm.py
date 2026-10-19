"""
난독화 한글 리뷰 복원 - 12단계: LLM 없이 문맥으로 어절을 고르는 '빈칸 채우기' 모델

08~10 에서는 고정되지 않은 어절을 공개 LLM(Qwen)이 앞뒤 문맥으로 골랐다. 12 는 그 자리에
train 정답 리뷰만으로 처음부터 학습한 작은 Transformer 를 쓴다 (사전학습 모델·외부 데이터 없음).

학습: 깨끗한 리뷰에서 어절(한글 구간)을 가리고, 앞뒤 문맥으로 가려진 글자의 초·중·종성을 맞힌다.
      가린 어절의 앞 몇 글자를 드러낸 경우도 함께 학습 → 채점할 때 어절을 왼쪽부터 한 글자씩 채워 가며
      log P(후보 | 앞뒤 문맥) = Σ log P(j번째 글자 | 문맥, 후보의 앞 j-1 글자) 를 계산할 수 있다.
채점: anchor_infill.py --scorer infill 로 사용 (InfillScorer.score_spans).

사용 예 (Colab, GPU):
    !python -u infill_lm.py --mode train                 # train 90% 로 학습, val 로 best 저장 -> outputs/infill_best.pt
    !python -u infill_lm.py --mode train --full --epochs 40   # test 용: train 전체, 고정 epoch -> outputs/infill_full.pt
"""
import argparse
import math
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from neural_tagger import Tagger, split_train
from noisy_channel import N_CHO, N_JONG, N_JUNG, dec, is_hangul

PAD, UNK, MASK = 0, 1, 2
BASE = 3
N_HANGUL = 11172
RUN = re.compile(r"[가-힣]+")


class Vocab:
    def __init__(self, others):
        self.others = list(others)
        self.other = {c: BASE + N_HANGUL + i for i, c in enumerate(self.others)}
        self.size = BASE + N_HANGUL + len(self.others)

    def id(self, c):
        return BASE + ord(c) - 0xAC00 if is_hangul(c) else self.other.get(c, UNK)


def encode(vocab, text, masked=()):
    """글자 id + 자모 id(+1). 가린 위치는 글자 MASK, 자모 0."""
    n = len(text)
    c = np.zeros(n, np.int64)
    a = np.zeros(n, np.int64)
    b = np.zeros(n, np.int64)
    g = np.zeros(n, np.int64)
    masked = set(masked)
    for i, ch in enumerate(text):
        if i in masked:
            c[i] = MASK
        else:
            c[i] = vocab.id(ch)
            if is_hangul(ch):
                j = dec(ch)
                a[i], b[i], g[i] = j[0] + 1, j[1] + 1, j[2] + 1
    return c, a, b, g


def pick_mask(text, rng, p_word):
    """가릴 위치: 어절을 p_word 확률로 고르고(최소 1개), 절반은 앞 j 글자를 드러낸 채 나머지만 가린다."""
    runs = [m.span() for m in RUN.finditer(text)]
    if not runs:
        return []
    chosen = [r for r in runs if rng.random() < p_word] or [rng.choice(runs)]
    pos = []
    for s, e in chosen:
        j = rng.randint(0, e - s - 1) if rng.random() < 0.5 else 0
        pos += list(range(s + j, e))
    return pos


class MaskSet(Dataset):
    def __init__(self, texts, vocab, max_len, p_word, seed=None):
        self.texts, self.vocab, self.max_len, self.p_word = texts, vocab, max_len, p_word
        self.seed = seed  # 고정 seed 면 매번 같은 마스크 (검증용)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, k):
        rng = random.Random(self.seed * 100003 + k) if self.seed is not None else random
        t = self.texts[k]
        if len(t) > self.max_len:
            s = rng.randint(0, len(t) - self.max_len)
            t = t[s : s + self.max_len]
        pos = pick_mask(t, rng, self.p_word)
        c, a, b, g = encode(self.vocab, t, pos)
        ta = np.full(len(t), -100, np.int64)
        tb = np.full(len(t), -100, np.int64)
        tg = np.full(len(t), -100, np.int64)
        for i in pos:
            ta[i], tb[i], tg[i] = dec(t[i])
        return c, a, b, g, ta, tb, tg


def collate(batch):
    n = max(len(x[0]) for x in batch)
    out = []
    for k in range(len(batch[0])):
        arr = np.full((len(batch), n), -100 if k >= 4 else 0, np.int64)
        for i, x in enumerate(batch):
            arr[i, : len(x[k])] = x[k]
        out.append(torch.from_numpy(arr))
    return out


def make_model(n_char, a, device):
    return Tagger(n_char, d=a["d"], layers=a["layers"], heads=a["heads"], ff=a["d"] * 4, drop=a["dropout"],
                  max_pos=max(512, a["max_len"])).to(device)


@torch.no_grad()
def evaluate(model, dl, device):
    """가린 글자 기준 손실과 정확도(초·중·종성 모두 맞음)."""
    model.eval()
    tot, n, ok = 0.0, 0, 0
    for batch in dl:
        c, a, b, g, ta, tb, tg = [t.to(device) for t in batch]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            oa, ob, og = model(c, a, b, g)
        m = ta >= 0
        k = int(m.sum())
        loss = sum(F.cross_entropy(o.float()[m], t[m], reduction="sum") for o, t in ((oa, ta), (ob, tb), (og, tg)))
        tot += float(loss)
        n += k
        ok += int(((oa.argmax(-1) == ta) & (ob.argmax(-1) == tb) & (og.argmax(-1) == tg))[m].sum())
    return tot / max(n, 1), ok / max(n, 1)


def train(args, device):
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    tr, va = (full, None) if args.full else split_train(full, args.val_ratio, args.seed)
    clean = lambda s: [str(y).rstrip() for y in s if str(y).strip()]
    texts = clean(tr.output)
    # 비한글 문자 집합: train·test 입력에도 그대로 나오므로 함께 수집 (정답 정보 아님)
    others = sorted({ch for t in list(full.output) + list(full.input) + list(test.input) for ch in str(t)
                     if not is_hangul(ch)})
    vocab = Vocab(others)
    hp = {k: getattr(args, k) for k in ("d", "layers", "heads", "dropout", "max_len", "p_word")}
    model = make_model(vocab.size, hp, device)
    ds = MaskSet(texts, vocab, args.max_len, args.p_word)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=args.loader_workers,
                    drop_last=True, persistent_workers=args.loader_workers > 0)
    vdl = None
    if va is not None:
        vdl = DataLoader(MaskSet(clean(va.output), vocab, args.max_len, args.p_word, seed=args.seed),
                         batch_size=256, collate_fn=collate)
    print(f"  params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | 학습 문장 {len(texts)} | steps/epoch {len(dl)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / max(1, int(0.05 * total))) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, "infill_full.pt" if args.full else "infill_best.pt")
    best, rows = 1e9, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for batch in dl:
            c, a, b, g, ta, tb, tg = [t.to(device, non_blocking=True) for t in batch]
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                outs = model(c, a, b, g)
            loss = sum(F.cross_entropy(o.float().transpose(1, 2), t, ignore_index=-100)
                       for o, t in zip(outs, (ta, tb, tg)))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        msg = f"epoch {ep:3d} | 학습 손실 {tot / len(dl):.4f} | {time.time() - t0:.0f}s"
        state = {"model": model.state_dict(), "hp": hp, "others": vocab.others, "epoch": ep, "full": args.full}
        vl = va_acc = None
        if vdl is not None and (ep % args.eval_every == 0 or ep == args.epochs):
            vl, va_acc = evaluate(model, vdl, device)
            msg += f" | 검증: 가린 글자 손실 {vl:.4f}, 정확도 {va_acc:.4f}"
            if vl < best:
                best = vl
                torch.save(state, ckpt)
                msg += " *"
        elif vdl is None:
            torch.save(state, ckpt)
        print(msg, flush=True)
        rows.append({"epoch": ep, "train_loss": round(tot / len(dl), 5), "val_loss": vl, "val_acc": va_acc,
                     **hp, "lr": args.lr, "epochs": args.epochs, "full": args.full})
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"infill_runs_{'full' if args.full else 'split'}.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"저장: {ckpt}" + (f" (검증 손실 최저 {best:.4f})" if vdl is not None else ""))


# --------------------------------------------------------------------------- #
# 채점기 (anchor_infill.py --scorer infill)
# --------------------------------------------------------------------------- #
class InfillScorer:
    """score_spans([(왼쪽 문맥, [후보...], 오른쪽 문맥), ...]) -> 후보별 log P(후보 | 앞뒤 문맥).
    어절을 왼쪽부터 한 글자씩 채운다: j번째 글자는 '후보의 앞 j 글자 + 나머지 가림' 상태에서 예측.
    앞 글자가 같은 후보들은 계산을 공유하고, 같은 문맥·앞부분은 캐시한다."""

    def __init__(self, ckpt, batch_tokens=65536):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st = torch.load(ckpt, map_location=self.device)
        self.vocab = Vocab(st["others"])
        self.model = make_model(self.vocab.size, st["hp"], self.device)
        self.model.load_state_dict(st["model"])
        self.model.eval()
        self.batch_tokens, self.cache, self.n_forward = batch_tokens, {}, 0
        print(f"  빈칸 채우기 모델 {ckpt} (epoch {st['epoch']}, "
              f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M) on {self.device}")

    @torch.no_grad()
    def _run(self, reqs):
        """reqs: [(key, 왼쪽, 드러낸 앞부분, 가린 글자 수, 오른쪽)] -> cache[key] = (초, 중, 종 log-prob 벡터)"""
        reqs = sorted(reqs, key=lambda r: len(r[1]) + len(r[2]) + r[3] + len(r[4]))
        i = 0
        while i < len(reqs):
            last = reqs[min(i + 1023, len(reqs) - 1)]
            L = len(last[1]) + len(last[2]) + last[3] + len(last[4])
            chunk = reqs[i : i + max(1, min(1024, self.batch_tokens // max(L, 1)))]
            items, pos = [], []
            for _, left, pre, nm, right in chunk:
                s = len(left) + len(pre)
                text = left + pre + "가" * nm + right  # 가린 자리는 자리표시자 (encode 에서 MASK 로 바뀜)
                items.append(encode(self.vocab, text, range(s, s + nm)))
                pos.append(s)
            c, a, b, g = [t.to(self.device) for t in collate(items)]
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                oa, ob, og = self.model(c, a, b, g)
            idx = torch.arange(len(chunk), device=self.device)
            p = torch.tensor(pos, device=self.device)
            la = F.log_softmax(oa.float()[idx, p], -1).cpu().numpy()
            lb = F.log_softmax(ob.float()[idx, p], -1).cpu().numpy()
            lg = F.log_softmax(og.float()[idx, p], -1).cpu().numpy()
            for r, req in enumerate(chunk):
                self.cache[req[0]] = (la[r], lb[r], lg[r])
            self.n_forward += len(chunk)
            i += len(chunk)

    def score_spans(self, groups):
        need = {}
        for left, cands, right in groups:
            for cand in cands:
                for j in range(len(cand)):
                    key = (left, cand[:j], len(cand), right)
                    if key not in self.cache and key not in need:
                        need[key] = (key, left, cand[:j], len(cand) - j, right)
        if need:
            self._run(list(need.values()))
        out = []
        for left, cands, right in groups:
            sc = []
            for cand in cands:
                s = 0.0
                for j, ch in enumerate(cand):
                    la, lb, lg = self.cache[(left, cand[:j], len(cand), right)]
                    x, y, z = dec(ch)
                    s += float(la[x] + lb[y] + lg[z])
                sc.append(s)
            out.append(sc)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train"], default="train")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--full", action="store_true", help="train 전체로 학습 (test 예측용, 고정 epoch)")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p_word", type=float, default=0.2, help="어절을 가릴 확률")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval_every", type=int, default=2)
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
