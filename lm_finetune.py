"""
난독화 한글 리뷰 복원 - 공개 언어모델을 train 리뷰로 미세조정 (LoRA)

08 의 문맥 판단은 공개 언어모델(Qwen2.5-1.5B)이 한다. 일반 한국어로 학습된 모델이라 숙박 리뷰의
구어·은어를 표준어 쪽으로 바꾸는 실수가 있다 (예: 지립니다 -> 질립니다, 굳은 -> 궂은).
제공된 train 의 정답 리뷰(output)만으로 가볍게 추가 학습시켜 리뷰 말투를 익히게 한다.
(외부 데이터 없음. 공개 가중치 모델 + 제공 데이터 → 대회 규칙 범위 안)

- 검증용(split): 검증 10% 를 뺀 train 90% 의 정답만 사용 → 08 검증 점수와 정직하게 비교 가능
- test 용(--full): train 전체 정답 사용
- 결과: LoRA 어댑터(작음, outputs/ 에 저장되어 드라이브 백업) + 병합 모델(약 3GB, models/ 에만 저장)
  병합 모델 폴더를 anchor_infill.py / lm_rerank.py 의 --lm 에 그대로 넣으면 된다.

사용 예 (Colab, GPU):
    !python -u lm_finetune.py                    # split 용 학습 + 병합 -> models/qwen_ft_split
    !python -u lm_finetune.py --full             # test 용 -> models/qwen_ft_full
    !python -u lm_finetune.py --merge_only       # 런타임이 바뀌었을 때: 저장된 어댑터로 병합 모델만 다시 만들기
"""
import argparse
import math
import os
import random
import time

import pandas as pd
import torch

from neural_tagger import split_train


def batches(ids_list, batch_tokens, shuffle, seed=0):
    """길이가 비슷한 문장끼리 묶은 배치 (토큰 수 상한)."""
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    out, cur, mx = [], [], 0
    for i in order:
        mx2 = max(mx, len(ids_list[i]))
        if cur and mx2 * (len(cur) + 1) > batch_tokens:
            out.append(cur)
            cur, mx2 = [], len(ids_list[i])
        cur.append(i)
        mx = mx2
    if cur:
        out.append(cur)
    if shuffle:
        random.Random(seed).shuffle(out)
    return out


def collate(ids_list, idx, pad, device):
    n = max(len(ids_list[i]) for i in idx)
    ids = torch.full((len(idx), n), pad, dtype=torch.long)
    att = torch.zeros((len(idx), n), dtype=torch.long)
    for r, i in enumerate(idx):
        ids[r, : len(ids_list[i])] = torch.tensor(ids_list[i])
        att[r, : len(ids_list[i])] = 1
    labels = ids.masked_fill(att == 0, -100)
    return ids.to(device), att.to(device), labels.to(device)


@torch.no_grad()
def eval_loss(model, ids_list, pad, device, batch_tokens, amp):
    model.eval()
    tot, n = 0.0, 0
    for idx in batches(ids_list, batch_tokens, False):
        ids, att, labels = collate(ids_list, idx, pad, device)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            loss = model(input_ids=ids, attention_mask=att, labels=labels).loss
        k = int((labels[:, 1:] != -100).sum())
        tot += float(loss) * k
        n += k
    return tot / max(n, 1)


def paths(args):
    tag = "full" if args.full else "split"
    return os.path.join(args.out_dir, f"lm_adapter_{tag}"), os.path.join(args.model_dir, f"qwen_ft_{tag}")


def merge(args, device):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    adapter, merged = paths(args)
    if not os.path.exists(adapter):
        raise SystemExit(f"{adapter} 가 없습니다. 먼저 학습을 실행하세요.")
    base = AutoModelForCausalLM.from_pretrained(args.lm, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    os.makedirs(merged, exist_ok=True)
    model.save_pretrained(merged)
    AutoTokenizer.from_pretrained(args.lm).save_pretrained(merged)
    print(f"병합 모델 저장: {merged}  → anchor_infill.py --lm {merged}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--model_dir", default="models", help="병합 모델 저장 폴더 (용량이 커서 드라이브 백업 대상 아님)")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--full", action="store_true", help="train 전체로 학습 (test 예측용)")
    ap.add_argument("--merge_only", action="store_true")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank")
    ap.add_argument("--batch_tokens", type=int, default=1024,
                    help="학습 배치당 토큰 수. 어휘가 15만 개라 학습 시 메모리를 많이 씀 (L4 22GB 에서 4096 은 부족)")
    args = ap.parse_args()
    print(vars(args))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.merge_only:
        return merge(args, device)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise SystemExit("peft 가 없습니다. 먼저 실행: !pip -q install peft")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    tr, va = (full, None) if args.full else split_train(full, args.val_ratio, args.seed)
    tok = AutoTokenizer.from_pretrained(args.lm)
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # 08 의 점수 계산과 같은 형식: [시작 토큰] + 리뷰 본문
    enc = lambda texts: [[start] + tok(str(t).rstrip(), add_special_tokens=False).input_ids[:512] for t in texts]
    tr_ids = enc(tr.output)
    va_ids = enc(va.output) if va is not None else None

    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = torch.bfloat16 if bf16 else (torch.float16 if device.type == "cuda" else None)
    model = AutoModelForCausalLM.from_pretrained(args.lm, torch_dtype=torch.float32).to(device)
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  학습 문장 {len(tr_ids)} | 학습 파라미터 {n_train / 1e6:.1f}M (전체의 "
          f"{n_train / sum(p.numel() for p in model.parameters()):.2%}) | amp {amp}")
    if va_ids:
        print(f"  [학습 전] 검증 리뷰 손실 {eval_loss(model, va_ids, pad, device, args.batch_tokens, amp):.4f}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(batches(tr_ids, args.batch_tokens, False))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * steps))) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp is torch.float16)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for idx in batches(tr_ids, args.batch_tokens, True, args.seed + ep):
            ids, att, labels = collate(tr_ids, idx, pad, device)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                loss = model(input_ids=ids, attention_mask=att, labels=labels).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss)
            n += 1
        msg = f"epoch {ep} | 학습 손실 {tot / n:.4f} | {time.time() - t0:.0f}s"
        if va_ids:
            msg += f" | 검증 리뷰 손실 {eval_loss(model, va_ids, pad, device, args.batch_tokens, amp):.4f}"
        print(msg, flush=True)

    adapter, _ = paths(args)
    model.save_pretrained(adapter)
    print(f"어댑터 저장: {adapter}")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    merge(args, device)


if __name__ == "__main__":
    main()
