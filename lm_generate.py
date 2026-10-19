"""
난독화 한글 리뷰 복원 - 10단계: 미세조정한 공개 언어모델로 리뷰 문장 생성 (어휘 증강)

09 에 남은 오류의 대부분은 01(노이즈 채널)이 모르는 단어다 (정답이 후보에 없는 오류의 89% 가 train 에 없는 어절:
반납하고·기념일·혼잡하다·찝찝하고·귀중품 등 흔한 말인데 train 리뷰 1만 개에 없었음).
train 리뷰로 미세조정한 Qwen2.5-1.5B 는 리뷰 말투 + 원래 알던 한국어 어휘를 함께 갖고 있으므로,
이 모델이 생성한 리뷰를 01 의 사전과 문자 n-gram 에 넣어 어휘를 넓힌다.
(외부 데이터가 아니라, 공개 가중치 모델 + 제공된 train 으로 만든 증강)

생성 방식 두 가지를 섞는다:
  - 이어 쓰기: train 리뷰의 첫 1~3 어절을 주고 뒤를 생성 (실제 리뷰의 시작을 따르므로 주제가 고름)
  - 자유 생성: 아무것도 주지 않고 생성
검증용(split)은 검증 10% 를 뺀 train 90% 로 미세조정한 모델과 그 90% 의 문장만 사용 → 검증 정답 누출 없음.

사용 예 (Colab, GPU):
    !python -u lm_generate.py                          # models/qwen_ft_split -> outputs/gen_reviews_split.txt
    !python -u lm_generate.py --full                   # models/qwen_ft_full  -> outputs/gen_reviews_full.txt
"""
import argparse
import os
import random
import re
import time
from collections import Counter

import pandas as pd
import torch

from neural_tagger import split_train

# 허용 문자: 한글 음절·자모, 영문·숫자, 흔한 문장부호. 그 외(한자·일본어·이모지 등)가 섞인 문장은 버린다.
ALLOWED = re.compile(r"^[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9\s.,!?~^()\[\]{}<>:;'\"%&*+\-_/=@#…·♡♥★☆]+$")
HANGUL_RUN = re.compile(r"[가-힣]+")


def clean(text, min_chars):
    """마지막 덜 끝난 어절을 잘라내고, 이상한 문자가 섞였거나 너무 짧은 문장은 버린다."""
    text = re.sub(r"\s+", " ", text.replace("�", "")).strip()
    if " " in text:
        text = text[: text.rfind(" ")]  # 토큰 수 제한으로 잘린 마지막 어절 제거
    if len(text) < min_chars or not ALLOWED.match(text):
        return None
    if len(HANGUL_RUN.findall(text)) < 3:
        return None
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--full", action="store_true", help="train 전체로 미세조정한 모델 사용 (test 예측용)")
    ap.add_argument("--lm", default="", help="생성에 쓸 모델 폴더 (기본: models/qwen_ft_split 또는 _full)")
    ap.add_argument("--n", type=int, default=30000, help="생성할 문장 수 (정리 후 남는 수는 이보다 적음)")
    ap.add_argument("--free_ratio", type=float, default=0.3, help="자유 생성 비율 (나머지는 train 리뷰 첫 어절 이어 쓰기)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--min_chars", type=int, default=12)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tag = "full" if args.full else "split"
    lm = args.lm or os.path.join(args.model_dir, f"qwen_ft_{tag}")
    print(vars(args), "| model", lm)
    if not os.path.exists(lm):
        raise SystemExit(f"{lm} 가 없습니다. 먼저: !python -u lm_finetune.py {'--full' if args.full else ''} "
                         f"(어댑터가 있으면 --merge_only)")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(lm)
    model = AutoModelForCausalLM.from_pretrained(
        lm, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    tr, va = (full, None) if args.full else split_train(full, args.val_ratio, args.seed)
    reviews = [str(y).strip() for y in tr.output if str(y).strip()]
    train_words = Counter(w for y in reviews for w in HANGUL_RUN.findall(y))

    def prompt():
        if random.random() < args.free_ratio:
            return ""
        words = random.choice(reviews).split(" ")
        return " ".join(words[: random.randint(1, 3)])

    out, seen, t0, made = [], set(reviews), time.time(), 0
    while made < args.n:
        prompts = [prompt() for _ in range(min(args.batch, args.n - made))]
        enc = [[start] + tok(p, add_special_tokens=False).input_ids for p in prompts]
        m = max(len(e) for e in enc)
        ids = torch.full((len(enc), m), pad, dtype=torch.long)
        att = torch.zeros((len(enc), m), dtype=torch.long)
        for r, e in enumerate(enc):  # 왼쪽 패딩 (생성은 오른쪽 끝에서 이어지므로)
            ids[r, m - len(e):] = torch.tensor(e)
            att[r, m - len(e):] = 1
        with torch.no_grad():
            gen = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), do_sample=True,
                                 temperature=args.temperature, top_p=args.top_p,
                                 max_new_tokens=args.max_new_tokens, pad_token_id=pad)
        for r, e in enumerate(enc):
            text = tok.decode(e[1:] + gen[r, m:].tolist(), skip_special_tokens=True)
            text = text.split("\n")[0]  # 리뷰 한 건만
            c = clean(text, args.min_chars)
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        made += len(prompts)
        if (made // args.batch) % 10 == 0 or made >= args.n:
            print(f"  생성 {made}/{args.n} | 남은 문장 {len(out)} | {time.time() - t0:.0f}s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"gen_reviews_{tag}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    gen_words = Counter(w for y in out for w in HANGUL_RUN.findall(y) if len(w) >= 2)
    new2 = {w: c for w, c in gen_words.items() if w not in train_words and c >= 2}
    print(f"저장: {path} | 문장 {len(out)}개 | 서로 다른 어절 {len(gen_words)} | "
          f"train 에 없고 2회 이상 나온 새 어절 {len(new2)}개")
    print("  새 어절 예 (많이 나온 순):", [w for w, _ in Counter(new2).most_common(40)])
    print("  생성 문장 예:")
    for y in random.sample(out, min(8, len(out))):
        print("   ", y[:100])
    if va is not None:  # 검증 정답 중 train 에 없던 어절을 얼마나 새로 덮는가 (증강 효과의 선행 지표)
        val_words = Counter(w for y in va.output for w in HANGUL_RUN.findall(str(y)) if len(w) >= 2)
        oov = {w: c for w, c in val_words.items() if w not in train_words}
        hit = sum(c for w, c in oov.items() if w in new2)
        print(f"  검증 정답 어절 중 train 에 없는 것 {sum(oov.values())}개 → 그중 생성 문장(2회 이상)이 덮는 것 "
              f"{hit}개 ({hit / max(sum(oov.values()), 1):.1%})")


if __name__ == "__main__":
    main()
