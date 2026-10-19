"""
난독화 한글 리뷰 복원 - 13단계: 글자 단위 사전학습 인코더 (KoCharELECTRA, Apache-2.0) 로 문맥 판단

LLM 은 아니지만 외부 사전학습 모델이다 (12 는 외부 모델 없음, 07~11 은 LLM).
KoCharELECTRA 는 음절 단위 사전(11,568개, 한글 음절 11,172개 전부 포함)이라 우리 글자 단위 채점과 그대로 맞는다.
전용 토크나이저 파일 없이 vocab.txt 만 받아 글자 -> id 로 직접 바꾼다.

두 모델을 채점기로 쓴다.
  generator      (ElectraForMaskedLM, 19M) : 후보 어절의 글자를 하나씩 가리고 맞히는 확률의 합 (pseudo log-likelihood)
  discriminator  (ElectraForPreTraining, 95M) : 후보를 끼운 문장에서 각 글자가 '원래 글자'로 보이는 정도 (바뀐 글자 탐지)

generator 는 train 정답 리뷰로 조금 더 학습시킬 수 있다 (--mode ft, 리뷰 말투 적응).

사용 예 (Colab, GPU):
    !python -u electra13.py --mode ft            # train 90% -> outputs/kochar_gen_split
    !python -u electra13.py --mode ft --full     # test 용: train 전체 -> outputs/kochar_gen_full
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

from neural_tagger import split_train

GEN = "monologg/kocharelectra-base-generator"
DISC = "monologg/kocharelectra-base-discriminator"
PAD, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4
RUN = re.compile(r"[가-힣]+")
UNK_LOGP = math.log(1e-6)  # 사전에 없는 글자가 든 후보의 글자당 점수


def load_vocab():
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(GEN, "vocab.txt"), encoding="utf-8") as f:
        toks = f.read().split("\n")
    c2i = {t: i for i, t in enumerate(toks) if len(t) == 1}
    assert toks[:5] == ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] and c2i[" "] == 5, "vocab.txt 형식이 예상과 다름"
    return c2i


def to_ids(c2i, text):
    return [CLS] + [c2i.get(c, UNK) for c in text] + [SEP]


def pad_batch(seqs, device):
    n = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), n), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, : len(s)] = torch.tensor(s)
    ids = ids.to(device)
    return ids, ids.ne(PAD).long()


# --------------------------------------------------------------------------- #
# generator 미세조정 (train 정답 리뷰, 어절 가리기)
# --------------------------------------------------------------------------- #
def mask_sample(ids, text, rng, p_word, n_vocab):
    """어절을 p_word 확률로 골라 절반은 어절 전체, 절반은 그 안의 한 글자만 가린다 (채점은 한 글자씩 가리므로)."""
    runs = [m.span() for m in RUN.finditer(text)]
    x, y = list(ids), [-100] * len(ids)
    if not runs:
        return x, y
    for s, e in [r for r in runs if rng.random() < p_word] or [rng.choice(runs)]:
        pos = range(s, e) if rng.random() < 0.5 else [rng.randrange(s, e)]
        for p in pos:
            y[p + 1] = ids[p + 1]  # +1: [CLS]
            r = rng.random()
            x[p + 1] = MASK if r < 0.8 else (rng.randrange(5, n_vocab) if r < 0.9 else ids[p + 1])
    return x, y


def finetune(args, device):
    from transformers import ElectraForMaskedLM
    c2i = load_vocab()
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    tr, va = (full, None) if args.full else split_train(full, args.val_ratio, args.seed)
    clean = lambda s: [str(y)[: args.max_len] for y in s if str(y).strip()]
    texts = clean(tr.output)
    model = ElectraForMaskedLM.from_pretrained(GEN).to(device)
    nv = model.config.vocab_size
    print(f"  generator {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | 학습 문장 {len(texts)}")

    def batches(ts, rng, shuffle):
        order = list(range(len(ts)))
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order), args.batch):
            xs, ys = zip(*[mask_sample(to_ids(c2i, ts[k]), ts[k], rng, args.p_word, nv) for k in order[i : i + args.batch]])
            x, att = pad_batch(xs, device)
            y = torch.full_like(x, -100)
            for r, yy in enumerate(ys):
                y[r, : len(yy)] = torch.tensor(yy)
            yield x, att, y

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = n = ok = 0
        for x, att, y in batches(clean(va.output), random.Random(args.seed), False):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                lg = model(input_ids=x, attention_mask=att).logits.float()
            m = y >= 0
            tot += float(F.cross_entropy(lg[m], y[m], reduction="sum"))
            ok += int((lg[m].argmax(-1) == y[m]).sum())
            n += int(m.sum())
        return tot / max(n, 1), ok / max(n, 1)

    out = os.path.join(args.out_dir, "kochar_gen_full" if args.full else "kochar_gen_split")
    if va is not None:
        vl, acc = evaluate()
        print(f"  미세조정 전 | 검증: 가린 글자 손실 {vl:.4f}, 정확도 {acc:.4f}")
    steps = args.epochs * math.ceil(len(texts) / args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, int(0.06 * steps))) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng, best, rows = random.Random(args.seed), 1e9, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, att, y in batches(texts, rng, True):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                lg = model(input_ids=x, attention_mask=att).logits
            loss = F.cross_entropy(lg.float().transpose(1, 2), y, ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        msg = f"epoch {ep:2d} | 학습 손실 {tot / nb:.4f} | {time.time() - t0:.0f}s"
        vl = acc = None
        if va is not None:
            vl, acc = evaluate()
            msg += f" | 검증: 가린 글자 손실 {vl:.4f}, 정확도 {acc:.4f}"
            if vl < best:
                best = vl
                model.save_pretrained(out)
                msg += " *"
        else:
            model.save_pretrained(out)
        print(msg, flush=True)
        rows.append({"epoch": ep, "train_loss": round(tot / nb, 5), "val_loss": vl, "val_acc": acc, "lr": args.lr,
                     "epochs": args.epochs, "batch": args.batch, "p_word": args.p_word, "full": args.full})
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"kochar_ft_runs_{'full' if args.full else 'split'}.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"저장: {out}")


# --------------------------------------------------------------------------- #
# 채점기
# --------------------------------------------------------------------------- #
class MLMScorer:
    """(문자열, s, e) -> 구간 [s, e) 의 글자를 하나씩 가리고 맞힌 log P 의 합. 같은 요청은 캐시."""

    def __init__(self, name, batch_tokens=65536):
        from transformers import ElectraForMaskedLM
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.c2i = load_vocab()
        self.model = ElectraForMaskedLM.from_pretrained(name).to(self.device).eval()
        self.batch_tokens, self.cache, self.n_forward = batch_tokens, {}, 0
        print(f"  MLM {name} ({sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M) on {self.device}")

    @torch.no_grad()
    def __call__(self, reqs):
        todo = sorted({r for r in reqs if r not in self.cache}, key=lambda r: len(r[0]))
        rows = [(r, j) for r in todo for j in range(r[1], r[2])]  # 요청마다 글자 수만큼
        acc = {r: 0.0 for r in todo}
        i = 0
        while i < len(rows):
            L = len(rows[min(i + 511, len(rows) - 1)][0][0]) + 2
            chunk = rows[i : i + max(1, min(512, self.batch_tokens // L))]
            seqs, tgt = [], []
            for (text, _, _), j in chunk:
                ids = to_ids(self.c2i, text)
                tgt.append(ids[j + 1])
                ids[j + 1] = MASK
                seqs.append(ids)
            x, att = pad_batch(seqs, self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                lg = self.model(input_ids=x, attention_mask=att).logits
            pos = torch.tensor([j + 1 for _, j in chunk], device=self.device)
            lp = F.log_softmax(lg[torch.arange(len(chunk), device=self.device), pos].float(), -1)
            lp = lp.gather(-1, torch.tensor(tgt, device=self.device)[:, None]).squeeze(-1).cpu().tolist()
            for (r, _), t, v in zip(chunk, tgt, lp):
                acc[r] += UNK_LOGP if t == UNK else v
            self.n_forward += len(chunk)
            i += len(chunk)
        self.cache.update(acc)
        return [self.cache[r] for r in reqs]


class RTDScorer:
    """(문자열, s, e) -> 구간 [s - pad, e + pad) 글자들의 '원래 글자' 점수 합 (discriminator).
    점수는 -logit (logit > 0 = 바뀐 글자) 을 ±10 으로 자른 값. 확률(logsigmoid)로 바꾸면 대부분 0 에 붙어
    차이가 사라지므로 logit 을 그대로 쓴다. 틀린 글자가 있으면 주변 글자의 점수도 함께 내려간다."""

    def __init__(self, name=DISC, batch_tokens=32768, pad=3):
        from transformers import ElectraForPreTraining
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.c2i = load_vocab()
        self.model = ElectraForPreTraining.from_pretrained(name).to(self.device).eval()
        self.batch_tokens, self.pad, self.cache, self.n_forward = batch_tokens, pad, {}, 0
        print(f"  RTD {name} ({sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M) on {self.device}")

    @torch.no_grad()
    def __call__(self, reqs):
        todo = sorted({r for r in reqs if r not in self.cache}, key=lambda r: len(r[0]))
        i = 0
        while i < len(todo):
            L = len(todo[min(i + 255, len(todo) - 1)][0]) + 2
            chunk = todo[i : i + max(1, min(256, self.batch_tokens // L))]
            x, att = pad_batch([to_ids(self.c2i, t) for t, _, _ in chunk], self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                lg = self.model(input_ids=x, attention_mask=att).logits.float()
            ok = (-lg).clamp(-10, 10).cpu().numpy()
            for r, (t, s, e) in enumerate(chunk):
                lo, hi = max(0, s - self.pad), min(len(t), e + self.pad)
                self.cache[(t, s, e)] = float(ok[r, lo + 1 : hi + 1].sum())
            self.n_forward += len(chunk)
            i += len(chunk)
        return [self.cache[r] for r in reqs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ft"], default="ft")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--full", action="store_true", help="train 전체로 미세조정 (test 예측용)")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=400)
    ap.add_argument("--p_word", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(vars(args), "| device", device)
    finetune(args, device)


if __name__ == "__main__":
    main()
