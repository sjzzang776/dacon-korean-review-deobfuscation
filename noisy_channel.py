"""
난독화 한글 리뷰 복원 - 노이즈 채널 베이스라인

    y* = argmax_y  lm_weight * log P_LM(y) + log P_ch(x | y)

- P_ch : 자모(초성/중성/종성) 분해 채널. 원본 초성이 ㅇ이면 직전 원본 글자의 받침(연음)에 조건부.
- P_LM : train output으로 학습한 문자 n-gram (interpolated Kneser-Ney).
- 디코딩: 글자별 후보 격자 위 빔 서치. 한글이 아닌 문자는 그대로 통과(train에서 100% 보존).
- 보조: 어절 사전(WordDict). 입력 어절과 중성 골격이 같은 train 어절들의 사후확률을
        글자별 보너스(dict_weight)로 빔 서치에 더한다. 사전에 없는 어절이면 보너스가 사라진다.
- 출력 길이 = 입력 길이 (평가가 같은 위치 문자 일치 기반이므로).

사용 예 (Colab):
    !python noisy_channel.py --mode val      # 현재 설정으로 검증 (outputs/nc_runs.csv 에 기록)
    !python noisy_channel.py --mode sweep    # 조합 탐색 -> outputs/best_params.json
    !python noisy_channel.py --mode test     # best_params.json 이 있으면 그 설정으로 제출 파일 생성
"""
import argparse
import itertools
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import pandas as pd

N_CHO, N_JUNG, N_JONG = 19, 21, 28
IEUNG = 11  # 초성 ㅇ 인덱스
BOS, EOS = "\x02", "\x03"


def is_hangul(c):
    return "가" <= c <= "힣"


def dec(c):
    o = ord(c) - 0xAC00
    return o // 588, (o % 588) // 28, o % 28


def comp(a, b, c):
    return chr(0xAC00 + a * 588 + b * 28 + c)


# 중성 골격: 난독화가 이 그룹 안에서만 모음을 바꾼다(train 일치율 98.8%).
# ㅔ→ㅞ, ㅚ→ㅞ 가 모두 있어서 ㅐ/ㅔ 계열과 ㅚ/ㅙ/ㅞ 계열은 한 그룹으로 합침.
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JUNG_CLASS = list(range(N_JUNG))
for _g in ["ㅏㅑ", "ㅓㅕㅝ", "ㅗㅛ", "ㅜㅠㅡ", "ㅣㅟㅢ", "ㅐㅔㅖㅒㅚㅙㅞ"]:
    for _c in _g:
        JUNG_CLASS[JUNG.index(_c)] = JUNG.index(_g[0])


def hangul_runs(s):
    """연속된 한글 구간 목록 [(시작 위치, 문자열)]. 공백·문장부호에서 끊김."""
    runs, i = [], 0
    while i < len(s):
        if is_hangul(s[i]):
            j = i
            while j < len(s) and is_hangul(s[j]):
                j += 1
            runs.append((i, s[i:j]))
            i = j
        else:
            i += 1
    return runs


def aligned_rows(df):
    """끝 공백을 제거한 뒤 길이가 같은 (input, output) 쌍만 사용."""
    rows = []
    for x, y in zip(df.input, df.output):
        x, y = x.rstrip(), y.rstrip()
        if len(x) == len(y):
            rows.append((x, y))
    return rows


# --------------------------------------------------------------------------- #
# 평가 (대회 공식: 같은 위치에서 일치한 문자 수 기반 F1, 행 평균)
# --------------------------------------------------------------------------- #
def f1_score(pred, gt):
    same = sum(a == b for a, b in zip(pred, gt))
    if same == 0:
        return 0.0
    p, r = same / len(pred), same / len(gt)
    return 2 * p * r / (p + r)


def mean_f1(preds, gts):
    return sum(f1_score(p, g) for p, g in zip(preds, gts)) / len(gts)


# --------------------------------------------------------------------------- #
# 문자 n-gram LM (interpolated Kneser-Ney)
# --------------------------------------------------------------------------- #
class KNLM:
    def __init__(self, texts, order=5, discount=0.75, cache_size=3_000_000):
        self.n, self.d, self.cache_size = order, discount, cache_size
        # counts[k] : (길이 k 문맥 + 다음 글자) -> 카운트. k = n-1 은 실제 카운트, 그 아래는 continuation 카운트.
        raw = Counter()
        for t in texts:
            s = BOS * (order - 1) + t + EOS
            for i in range(order - 1, len(s)):
                raw[s[i - order + 1 : i + 1]] += 1
        self.counts = [None] * order
        self.counts[order - 1] = raw
        for k in range(order - 2, -1, -1):
            cont = Counter()
            for key in self.counts[k + 1]:
                cont[key[1:]] += 1
            self.counts[k] = cont
        # stats[k][ctx] = [합계, 서로 다른 다음 글자 수]
        self.stats = []
        for k in range(order):
            st = defaultdict(lambda: [0, 0])
            for key, c in self.counts[k].items():
                e = st[key[:-1]]
                e[0] += c
                e[1] += 1
            self.stats.append(dict(st))
        self.vocab = {key[-1] for key in self.counts[0]} - {BOS}
        self.V = len(self.vocab) + 1
        self.cache = {}

    def logp(self, h, w):
        """h: 직전 n-1 글자 문맥 문자열."""
        key = h + w
        v = self.cache.get(key)
        if v is not None:
            return v
        p = 1.0 / self.V
        for k in range(min(self.n, len(h) + 1)):
            ctx = h[len(h) - k :]
            st = self.stats[k].get(ctx)
            if st is None:
                break  # 더 긴 문맥도 본 적 없음
            tot, types = st
            c = self.counts[k].get(ctx + w, 0)
            p = max(c - self.d, 0) / tot + self.d * types / tot * p
        v = math.log(p)
        if len(self.cache) > self.cache_size:
            self.cache.clear()
        self.cache[key] = v
        return v


# --------------------------------------------------------------------------- #
# 자모 분해 노이즈 채널
# --------------------------------------------------------------------------- #
class JamoChannel:
    """
    받침 탈락은 다음 글자에서 연음이 일어날 때만 생긴다(train: 연음 없으면 탈락 0%).
    그래서 받침 채널을 다음 원본 글자에 따라 둘로 나눈다.
      - 다음 원본 초성이 ㅇ : (앞 글자 난독 받침, 다음 글자 난독 초성)을 결합 분포 lp_pair[J] 로 함께 점수화
      - 그 외(다음 초성이 ㅇ 아님, 또는 한글 구간 끝) : lp_jong_plain[J] (연음 탈락이 섞이지 않은 받침 분포)
    use_pair=False 이면 예전 독립 채널(전체 평균 받침 분포 + 연음 조건부 초성)로 동작한다(비교용).
    """

    def __init__(self, rows, alpha=0.1, beta=5.0, beta_pair=20.0, cand_thr=0.005, use_pair=True):
        self.use_pair = use_pair
        cho_plain = defaultdict(Counter)   # y초성 -> x초성
        cho_ieung = defaultdict(Counter)   # (원본 초성 ㅇ) 직전 원본 받침 -> x초성
        jung = defaultdict(Counter)
        jong = defaultdict(Counter)        # 전체 평균 받침 분포 (후보 정렬·비교용)
        jong_plain = defaultdict(Counter)  # 다음 원본 초성이 ㅇ 이 아닐 때의 받침 분포
        pair = defaultdict(Counter)        # 원본 받침 J (다음 원본 초성 ㅇ) -> (앞 x받침, 다음 x초성)
        for x, y in rows:
            prev = None  # (x자모, y자모) of 직전 한글 글자
            for xc, yc in zip(x, y):
                if not (is_hangul(xc) and is_hangul(yc)):
                    if prev:
                        jong_plain[prev[1][2]][prev[0][2]] += 1
                    prev = None
                    continue
                xj, yj = dec(xc), dec(yc)
                pj = prev[1][2] if prev else 0
                if yj[0] == IEUNG:
                    cho_ieung[pj][xj[0]] += 1
                    if prev:
                        pair[pj][(prev[0][2], xj[0])] += 1
                elif prev:
                    jong_plain[pj][prev[0][2]] += 1
                cho_plain[yj[0]][xj[0]] += 1
                jung[yj[1]][xj[1]] += 1
                jong[yj[2]][xj[2]] += 1
                prev = (xj, yj)
            if prev:
                jong_plain[prev[1][2]][prev[0][2]] += 1

        def table(cnt, n_src, n_tgt):
            out = []
            for s in range(n_src):
                c = cnt[s]
                tot = sum(c.values())
                out.append([math.log((c[t] + alpha) / (tot + alpha * n_tgt)) for t in range(n_tgt)])
            return out

        self.lp_cho = table(cho_plain, N_CHO, N_CHO)
        self.lp_jung = table(jung, N_JUNG, N_JUNG)
        self.lp_jong = table(jong, N_JONG, N_JONG)
        self.lp_jong_plain = table(jong_plain, N_JONG, N_JONG)
        # 연음 조건부 초성(ㅇ) 분포: 받침별 카운트를 일반 ㅇ 분포 쪽으로 beta 만큼 수축
        base = [math.exp(v) for v in self.lp_cho[IEUNG]]
        self.lp_cho_ieung = []
        for pj in range(N_JONG):
            c = cho_ieung[pj]
            tot = sum(c.values())
            self.lp_cho_ieung.append(
                [math.log((c[t] + beta * base[t]) / (tot + beta)) for t in range(N_CHO)]
            )
        # 결합 분포: 독립 곱 P(x받침|J)·P(x초성|ㅇ,J) 쪽으로 beta_pair 만큼 수축. 인덱스 = x받침*N_CHO + x초성
        # J=0(원본 받침 없음)도 포함: 다음 글자가 ㅇ 일 때 새로 붙는 받침이 초성과 상관있을 수 있음
        self.lp_pair = []
        for J in range(N_JONG):
            c = pair[J]
            tot = sum(c.values())
            row = []
            for g in range(N_JONG):
                for t in range(N_CHO):
                    prior = math.exp(self.lp_jong[J][g] + self.lp_cho_ieung[J][t])
                    row.append(math.log((c[(g, t)] + beta_pair * prior) / (tot + beta_pair)))
            self.lp_pair.append(row)

        # 역방향 후보: x 자모를 만들 수 있는 y 자모 (x 자모 기준 점유율 >= cand_thr)
        def inverse(cnt, n_src, n_tgt):
            inv = []
            for t in range(n_tgt):
                col = {s: cnt[s][t] for s in range(n_src) if cnt[s][t] > 0}
                tot = sum(col.values())
                keep = [s for s, v in col.items() if v / tot >= cand_thr] if tot else []
                inv.append(keep or [t])
            return inv

        self.inv_cho = inverse(cho_plain, N_CHO, N_CHO)
        self.inv_jung = inverse(jung, N_JUNG, N_JUNG)
        self.inv_jong = inverse(jong, N_JONG, N_JONG)

    def candidates(self, xc, vocab):
        """(y, y초성, 중성 logp, 문맥 무관 전체 채널 logp). 받침 logp 는 디코더에서 다음 글자를 본 뒤 더한다."""
        xj = dec(xc)
        out = []
        for a in self.inv_cho[xj[0]]:
            for b in self.inv_jung[xj[1]]:
                for c in self.inv_jong[xj[2]]:
                    y = comp(a, b, c)
                    if y in vocab:
                        lj = self.lp_jung[b][xj[1]]
                        static = self.lp_cho[a][xj[0]] + lj + self.lp_jong[c][xj[2]]
                        out.append((y, a, lj, static))
        return out

    def link_logp(self, J, prev_xjong, ycho, xcho):
        """앞 글자 받침(원본 J, 난독 prev_xjong) + 현재 글자 초성(원본 ycho, 난독 xcho)."""
        if not self.use_pair:  # 예전 독립 채널
            c = self.lp_cho_ieung[J][xcho] if ycho == IEUNG else self.lp_cho[ycho][xcho]
            return self.lp_jong[J][prev_xjong] + c
        if ycho == IEUNG:
            return self.lp_pair[J][prev_xjong * N_CHO + xcho]
        return self.lp_jong_plain[J][prev_xjong] + self.lp_cho[ycho][xcho]

    def end_jong_logp(self, J, xjong):
        """한글 구간 마지막 글자의 받침 (뒤에 연음 없음)."""
        return (self.lp_jong_plain if self.use_pair else self.lp_jong)[J][xjong]

    def seq_logp(self, xs, ys):
        """한글 구간 전체의 채널 logp. 구간 앞뒤는 한글이 아니라고 가정(연음 없음)."""
        s = 0.0
        for i, (xc, yc) in enumerate(zip(xs, ys)):
            xj, yj = dec(xc), dec(yc)
            s += self.lp_jung[yj[1]][xj[1]]
            if i == 0:
                s += self.lp_cho_ieung[0][xj[0]] if yj[0] == IEUNG else self.lp_cho[yj[0]][xj[0]]
            else:
                s += self.link_logp(dec(ys[i - 1])[2], dec(xs[i - 1])[2], yj[0], xj[0])
        return s + self.end_jong_logp(dec(ys[-1])[2], dec(xs[-1])[2])


# --------------------------------------------------------------------------- #
# 어절 사전 (보조 점수)
# --------------------------------------------------------------------------- #
class WordDict:
    """
    train output 의 한글 구간(어절)을 중성 골격으로 색인.
    입력 구간 xr 에 대해  P(w | xr) ∝ count(w) · P_ch(xr | w)  를 사전 어절들 + '사전에 없음(OOV)' 위에서 정규화하고,
    위치별로 '그 글자를 가진 어절들의 사후확률 합'을 반환한다. OOV 확률이 크면 보너스도 작아진다.
    OOV 점수 = log(oov_count) + 글자별 최선 채널 logp 의 합 (가장 그럴듯한 임의 복원).
    """

    def __init__(self, texts, ch, vocab, min_len=2, max_cands=200, oov_count=1.0,
                 extra_texts=None, extra_weight=0.0, extra_min=2):
        self.ch, self.vocab, self.min_len = ch, vocab, min_len
        self.log_oov = math.log(oov_count)
        texts = list(texts)
        self.counts = Counter(w for t in texts for _, w in hangul_runs(t) if len(w) >= min_len)
        # 10단계 증강: 공개 언어모델이 생성한 리뷰에서 train 에 없던 어절만 추가 (기존 어절의 빈도는 건드리지 않음).
        # 빈도 = 생성 문장에서의 횟수를 train 문장 수 기준으로 환산한 값 x extra_weight. extra_min 회 미만은 버림(오타·잡음).
        self.n_extra = 0
        if extra_texts and extra_weight > 0:
            gen = Counter(w for t in extra_texts for _, w in hangul_runs(t) if len(w) >= min_len)
            scale = extra_weight * len(texts) / len(extra_texts)
            for w, c in gen.items():
                if c >= extra_min and w not in self.counts:
                    self.counts[w] = c * scale
                    self.n_extra += 1
        self.index = defaultdict(list)
        for w, c in self.counts.items():
            self.index[self.key(w)].append((c, w))
        for v in self.index.values():
            v.sort(reverse=True)
            del v[max_cands:]
        self.best_char = {}
        self.cache = {}

    @staticmethod
    def key(w):
        return tuple(JUNG_CLASS[dec(c)[1]] for c in w)

    def _best_char_logp(self, xc):
        v = self.best_char.get(xc)
        if v is None:
            cands = self.ch.candidates(xc, self.vocab)
            v = max((static for *_, static in cands), default=0.0)
            self.best_char[xc] = v
        return v

    def posterior(self, xr):
        """위치별 {원본 글자: 사후확률 합} 리스트. 매칭 사전 어절이 없으면 None."""
        if xr in self.cache:
            return self.cache[xr]
        res = None
        cands = self.index.get(self.key(xr))
        if cands:
            scored = [(math.log(c) + self.ch.seq_logp(xr, w), w) for c, w in cands]
            oov = self.log_oov + sum(self._best_char_logp(xc) for xc in xr)
            mx = max(oov, max(s for s, _ in scored))
            z = math.exp(oov - mx) + sum(math.exp(s - mx) for s, _ in scored)
            res = [defaultdict(float) for _ in xr]
            for s, w in scored:
                p = math.exp(s - mx) / z
                if p < 1e-4:
                    continue
                for i, c in enumerate(w):
                    res[i][c] += p
        self.cache[xr] = res
        return res

    def top1(self, xr):
        """사전만으로 복원(진단용). 매칭이 없으면 None."""
        post = self.posterior(xr)
        if not post:
            return None
        return "".join(max(d.items(), key=lambda kv: kv[1])[0] if d else xc for d, xc in zip(post, xr))


# --------------------------------------------------------------------------- #
# 디코더
# --------------------------------------------------------------------------- #
class Decoder:
    def __init__(self, lm, ch, beam=8, topk=12, lm_weight=1.0, wd=None, dict_weight=0.0, tagger_weight=0.0):
        self.lm, self.ch = lm, ch
        self.beam, self.topk, self.lm_weight = beam, topk, lm_weight
        self.wd, self.dict_weight = wd, dict_weight
        self.tagger_weight = tagger_weight  # decode(x, lp) 로 태거 log-prob 이 주어질 때만 사용 (04 앙상블)
        self.cand_cache = {}

    def dict_bonus(self, x):
        """위치 -> {원본 글자: 보너스}. 사전 어절이 가리키는 글자에 dict_weight * 사후확률 만큼 가점."""
        bonus = {}
        if self.wd is None or self.dict_weight <= 0:
            return bonus
        for i0, xr in hangul_runs(x):
            if len(xr) < self.wd.min_len:
                continue
            post = self.wd.posterior(xr)
            if not post:
                continue
            for j, d in enumerate(post):
                if d:
                    bonus[i0 + j] = {c: self.dict_weight * p for c, p in d.items()}
        return bonus

    def cands(self, xc):
        """(y, y초성, 중성 logp) 상위 topk. 문맥 무관 채널 + 유니그램 LM 으로 정렬."""
        v = self.cand_cache.get(xc)
        if v is None:
            scored = []
            for y, a, lj, static in self.ch.candidates(xc, self.lm.vocab):
                scored.append((static + self.lm_weight * self.lm.logp("", y), y, a, lj))
            scored.sort(reverse=True)
            v = [(y, a, lj) for _, y, a, lj in scored[: self.topk]]
            if not v:  # 후보가 없으면 입력 글자 그대로
                v = [(xc, dec(xc)[0], 0.0)]
            self.cand_cache[xc] = v
        return v

    def _pending_jong(self, h, px):
        """직전 글자 받침 채널 logp (다음 글자가 한글이 아니거나 문장 끝일 때 확정)."""
        if px and is_hangul(px) and is_hangul(h[-1]):
            return self.ch.end_jong_logp(dec(h[-1])[2], dec(px)[2])
        return 0.0

    def decode(self, x, lp=None):
        """lp: (초성, 중성, 종성) 위치별 log-prob 배열 3개 (neural_tagger.predict_logprobs 출력). 있으면
        tagger_weight * (log P초 + log P중 + log P종) 을 후보 점수에 더하고, 태거 1순위 글자를 후보에 추가."""
        beams, px = self._run(x, lp)
        w = self.lm_weight
        best = max(  # 마지막 글자 받침 + EOS 포함 최종 선택
            beams.items(),
            key=lambda kv: kv[1][0] + self._pending_jong(kv[0], px) + w * self.lm.logp(kv[0], EOS),
        )
        return self._backtrack(best[1][1])

    def decode_nbest(self, x, k, ctx_y="", ctx_x="", final=True):
        """상위 k개 후보 [(문자열, 점수)] (07 언어모델 재순위화용).
        ctx_y/ctx_x: 앞 조각의 복원 결과/입력 (n-gram 문맥과 연음 연결에 사용). final=False 면 EOS 점수 제외."""
        beams, px = self._run(x, None, ctx_y, ctx_x)
        w = self.lm_weight
        scored = []
        for h, (sc, node) in beams.items():
            s = sc + self._pending_jong(h, px) + (w * self.lm.logp(h, EOS) if final else 0.0)
            scored.append((s, node))
        scored.sort(key=lambda t: -t[0])
        return [(self._backtrack(node), s) for s, node in scored[:k]]

    @staticmethod
    def _backtrack(node):
        out = []
        while node is not None:
            out.append(node[0])
            node = node[1]
        return "".join(reversed(out))

    def _run(self, x, lp=None, ctx_y="", ctx_x=""):
        """빔 탐색 본체. (마지막 빔들, 마지막 입력 글자) 반환."""
        lm, ch, w = self.lm, self.ch, self.lm_weight
        use_tagger = lp is not None and self.tagger_weight > 0
        h0 = (BOS * (lm.n - 1) + ctx_y)[-(lm.n - 1):]
        beams = {h0: (0.0, None)}  # 문맥 -> (점수, 역추적 노드 (글자, 부모))
        px = ctx_x[-1:]  # 직전 입력 글자
        bonus = self.dict_bonus(x)
        for i, xc in enumerate(x):
            new = {}
            if is_hangul(xc):
                xj = dec(xc)
                xcho = xj[0]
                pxg = dec(px)[2] if px and is_hangul(px) else None
                cands = self.cands(xc)
                bo = bonus.get(i, {})
                want = list(bo)
                if use_tagger:
                    la, lb, lg = lp[0][i], lp[1][i], lp[2][i]
                    want.append(comp(int(la.argmax()), int(lb.argmax()), int(lg.argmax())))
                if want:  # 사전·태거가 가리키는 글자가 후보에 없으면 추가
                    have = {y for y, _, _ in cands}
                    extra = []
                    for y in want:
                        if y not in have and y in lm.vocab:
                            have.add(y)
                            yj = dec(y)
                            extra.append((y, yj[0], ch.lp_jung[yj[1]][xj[1]]))
                    if extra:
                        cands = cands + extra
                if use_tagger:
                    tw = self.tagger_weight
                    tsc = {}
                    for y, _, _ in cands:
                        a_, b_, g_ = dec(y)
                        tsc[y] = tw * float(la[a_] + lb[b_] + lg[g_])
                    if bo:
                        bo = {y: bo.get(y, 0.0) + tsc[y] for y in tsc}
                    else:
                        bo = tsc
                for h, (sc, node) in beams.items():
                    prev = h[-1]
                    linked = pxg is not None and is_hangul(prev)
                    J = dec(prev)[2] if linked else 0
                    for y, a, lj in cands:
                        if linked:  # 앞 받침 + 현재 초성 결합 점수
                            link = ch.link_logp(J, pxg, a, xcho)
                        elif a == IEUNG:
                            link = ch.lp_cho_ieung[0][xcho]
                        else:
                            link = ch.lp_cho[a][xcho]
                        s = sc + lj + link + w * lm.logp(h, y) + bo.get(y, 0.0)
                        nh = h[1:] + y
                        old = new.get(nh)
                        if old is None or s > old[0]:
                            new[nh] = (s, (y, node))
            else:
                for h, (sc, node) in beams.items():
                    s = sc + self._pending_jong(h, px) + w * lm.logp(h, xc)
                    nh = h[1:] + xc
                    old = new.get(nh)
                    if old is None or s > old[0]:
                        new[nh] = (s, (xc, node))
            if len(new) > self.beam:
                beams = dict(sorted(new.items(), key=lambda kv: -kv[1][0])[: self.beam])
            else:
                beams = new
            px = xc
        return beams, px


_DECODER = None


def _decode_one(item):
    return _DECODER.decode(*item)


def decode_all(decoder, texts, workers, lps=None):
    """lps: 문장별 태거 log-prob (04 앙상블). None 이면 노이즈 채널 단독."""
    global _DECODER
    _DECODER = decoder
    t = time.time()
    items = list(zip(texts, lps)) if lps is not None else [(x,) for x in texts]
    if workers > 1:
        # Colab(Linux)은 fork 이므로 모델이 자식 프로세스로 복사 없이 공유됨
        with Pool(workers) as pool:
            preds = list(pool.imap(_decode_one, items, chunksize=16))
    else:
        preds = [decoder.decode(*it) for it in items]
    print(f"  decoded {len(texts)} rows in {time.time() - t:.1f}s")
    return preds


def load_aug(args):
    """10단계 증강용 생성 문장 (한 줄에 한 문장). aug_file 이 없으면 빈 목록."""
    path = getattr(args, "aug_file", "") or ""
    if not path:
        return []
    if not os.path.exists(path):
        raise SystemExit(f"증강 파일이 없습니다: {path}")
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def build(train_df, args):
    rows = aligned_rows(train_df)
    t = time.time()
    extra = load_aug(args)
    n_lm = int(getattr(args, "aug_lm_n", 0) or 0)
    w_dict = float(getattr(args, "aug_dict_weight", 0.0) or 0.0)
    lm = KNLM([y for y in train_df.output] + extra[:n_lm], order=args.order)
    ch = JamoChannel(rows, cand_thr=args.cand_thr, use_pair=not args.no_pair)
    wd = WordDict(train_df.output, ch, lm.vocab, min_len=args.dict_min_len, oov_count=args.dict_oov,
                  extra_texts=extra, extra_weight=w_dict, extra_min=int(getattr(args, "aug_min_count", 2) or 2))
    print(
        f"  model built in {time.time() - t:.1f}s  (LM vocab={lm.V - 1}, aligned rows={len(rows)}, "
        f"dict words={len(wd.counts)})"
        + (f" | 증강: 생성 문장 {len(extra)}개 중 LM 에 {min(n_lm, len(extra))}개, 사전에 새 어절 {wd.n_extra}개"
           if extra else "")
    )
    decoder = Decoder(
        lm, ch, beam=args.beam, topk=args.topk, lm_weight=args.lm_weight,
        wd=wd, dict_weight=args.dict_weight,
    )
    return decoder, rows


def dict_report(wd, inputs, gts):
    """검증용: 정답 어절의 사전 포함률, 사전만으로 복원했을 때 어절 정확도."""
    n = in_dict = matched = top_ok = 0
    for x, g in zip(inputs, gts):
        for i0, xr in hangul_runs(x):
            if len(xr) < wd.min_len:
                continue
            gw = g[i0 : i0 + len(xr)]
            n += 1
            in_dict += gw in wd.counts
            t = wd.top1(xr)
            if t is not None:
                matched += 1
                top_ok += t == gw
    print(
        f"  dict: 정답 어절 사전 포함 {in_dict / n:.3f} | 매칭된 어절 {matched / n:.3f} | "
        f"매칭 시 사전 top1 정확도 {top_ok / max(matched, 1):.3f}  (어절 {n}개)"
    )


def lookup_baseline(rows, texts):
    """비교용: 입력 글자별 최빈 원본 글자로 치환."""
    m = defaultdict(Counter)
    for x, y in rows:
        for a, b in zip(x, y):
            m[a][b] += 1
    best = {a: c.most_common(1)[0][0] for a, c in m.items()}
    return ["".join(best.get(a, a) for a in x) for x in texts]


# --------------------------------------------------------------------------- #
# 실행 기록 / 하이퍼파라미터 탐색
# --------------------------------------------------------------------------- #
AUG_KEYS = ["aug_file", "aug_lm_n", "aug_dict_weight", "aug_min_count"]  # 10단계: 생성 문장 증강
PARAM_KEYS = ["order", "cand_thr", "no_pair", "dict_min_len", "dict_oov", "beam", "topk", "lm_weight",
              "dict_weight"] + AUG_KEYS
BUILD_KEYS = ["order", "cand_thr", "no_pair", "dict_min_len", "dict_oov"] + AUG_KEYS  # 바뀌면 모델을 다시 만들어야 함
DEFAULT_GRID = "order=5,6;lm_weight=1.1,1.3,1.5;beam=8,16;dict_oov=0.5,1,2"


def params_of(args):
    return {k: getattr(args, k) for k in PARAM_KEYS}


def parse_grid(spec, args):
    """'lm_weight=1.0,1.3;beam=8,16' -> {'lm_weight': [1.0, 1.3], 'beam': [8, 16]}"""
    grid = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        k, vs = (t.strip() for t in part.split("=", 1))
        if k not in PARAM_KEYS:
            raise SystemExit(f"탐색할 수 없는 파라미터: {k} (가능: {PARAM_KEYS})")
        cast = type(getattr(args, k))
        if cast is bool:
            grid[k] = [v.strip().lower() in ("1", "true", "yes") for v in vs.split(",")]
        else:
            grid[k] = [cast(v) for v in vs.split(",")]
    return grid


def log_run(out_dir, row):
    """모든 검증 실행을 outputs/nc_runs.csv 에 누적 기록."""
    path = os.path.join(out_dir, "nc_runs.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_best(out_dir, params, f1, n_val):
    """best_params.json 갱신 (같은 검증 크기에서 더 좋을 때만)."""
    path = os.path.join(out_dir, "best_params.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("n_val") == n_val and old.get("f1", -1) >= f1:
            print(f"  best_params.json 유지 (기존 F1 {old['f1']:.4f} >= {f1:.4f})")
            return
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"params": params, "f1": f1, "n_val": n_val,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=1)
    print(f"  best_params.json 저장: F1 {f1:.4f} {params}")


def split_val(train, args):
    idx = list(range(len(train)))
    random.Random(args.seed).shuffle(idx)
    n_val = int(len(idx) * args.val_ratio)
    va = train.iloc[idx[:n_val]].reset_index(drop=True)
    tr = train.iloc[idx[n_val:]].reset_index(drop=True)
    if args.val_limit:
        va = va.iloc[: args.val_limit]
    return tr, va


def save_val_pred(out_dir, va, preds):
    va.assign(pred=preds, f1=[f1_score(p, g) for p, g in zip(preds, va.output)]).to_csv(
        os.path.join(out_dir, "val_pred.csv"), index=False, encoding="utf-8-sig"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--mode", choices=["val", "sweep", "test", "both"], default="val")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--val_limit", type=int, default=0, help="검증 행 수 제한(빠른 실험용, 0=전체)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--cand_thr", type=float, default=0.005)
    ap.add_argument("--lm_weight", type=float, default=1.3)
    ap.add_argument("--dict_weight", type=float, default=3.0, help="어절 사전 보너스 크기 (0이면 사전 끔)")
    ap.add_argument("--dict_min_len", type=int, default=2, help="사전을 적용할 최소 어절 길이")
    ap.add_argument("--dict_oov", type=float, default=1.0, help="'사전에 없음' 가상 빈도. 클수록 사전을 덜 믿음")
    ap.add_argument("--no_pair", action="store_true", help="받침·연음 결합 채널 끄기 (예전 독립 채널, 비교용)")
    ap.add_argument("--aug_file", default="", help="10단계: 공개 언어모델이 생성한 리뷰 파일 (한 줄에 한 문장)")
    ap.add_argument("--aug_lm_n", type=int, default=0, help="문자 n-gram LM 에 추가할 생성 문장 수 (0 = 추가 안 함)")
    ap.add_argument("--aug_dict_weight", type=float, default=0.0, help="사전에 추가하는 새 어절의 빈도 가중치 (0 = 추가 안 함)")
    ap.add_argument("--aug_min_count", type=int, default=2, help="생성 문장에서 이 횟수 이상 나온 새 어절만 사전에 추가")
    ap.add_argument("--grid", default=DEFAULT_GRID, help="sweep 탐색 범위. 예: 'lm_weight=1.0,1.3;beam=8,16'")
    ap.add_argument("--ignore_best", action="store_true", help="test 때 best_params.json 무시하고 명령줄 값 사용")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    print(vars(args))
    os.makedirs(args.out_dir, exist_ok=True)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))

    if args.mode in ("val", "both"):
        print("[val]")
        tr, va = split_val(train, args)
        decoder, rows = build(tr, args)
        gts = list(va.output)
        print(f"  identity     F1 = {mean_f1(list(va.input), gts):.4f}")
        print(f"  lookup       F1 = {mean_f1(lookup_baseline(rows, va.input), gts):.4f}")
        dict_report(decoder.wd, list(va.input), gts)
        t = time.time()
        preds = decode_all(decoder, list(va.input), args.workers)
        f1 = mean_f1(preds, gts)
        print(f"  noisy-channel F1 = {f1:.4f}")
        log_run(args.out_dir, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "val", "n_val": len(va),
                               **params_of(args), "f1": round(f1, 5), "sec": round(time.time() - t, 1)})
        save_val_pred(args.out_dir, va, preds)

    if args.mode == "sweep":
        print("[sweep]")
        tr, va = split_val(train, args)
        gts = list(va.output)
        grid = parse_grid(args.grid, args)
        combos = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
        # 모델 구성 파라미터가 같은 조합끼리 묶어서 재사용
        combos.sort(key=lambda c: tuple(str(c.get(k, getattr(args, k))) for k in BUILD_KEYS))
        print(f"  {len(combos)} 조합 x 검증 {len(va)}행")
        built, decoder = None, None
        best_f1, best_params, best_preds = -1.0, None, None
        for i, combo in enumerate(combos, 1):
            a = argparse.Namespace(**{**vars(args), **combo})
            bk = tuple(getattr(a, k) for k in BUILD_KEYS)
            if bk != built:
                decoder, _ = build(tr, a)
                built = bk
            decoder.beam, decoder.topk = a.beam, a.topk
            decoder.lm_weight, decoder.dict_weight = a.lm_weight, a.dict_weight
            decoder.cand_cache.clear()  # topk·lm_weight 가 후보 정렬에 쓰임
            t = time.time()
            preds = decode_all(decoder, list(va.input), args.workers)
            f1 = mean_f1(preds, gts)
            log_run(args.out_dir, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "sweep", "n_val": len(va),
                                   **params_of(a), "f1": round(f1, 5), "sec": round(time.time() - t, 1)})
            mark = ""
            if f1 > best_f1:
                best_f1, best_params, best_preds = f1, params_of(a), preds
                mark = " *"
            print(f"  [{i}/{len(combos)}] {combo} -> F1 {f1:.4f}{mark}", flush=True)
        print(f"  sweep best F1 {best_f1:.4f} {best_params}")
        save_val_pred(args.out_dir, va, best_preds)
        save_best(args.out_dir, best_params, best_f1, len(va))

    if args.mode in ("test", "both"):
        print("[test]")
        best_path = os.path.join(args.out_dir, "best_params.json")
        source = "명령줄 값"
        if not args.ignore_best and os.path.exists(best_path):
            with open(best_path, encoding="utf-8") as f:
                best = json.load(f)
            cli_aug = args.aug_file  # test 용 생성 파일을 명령줄로 주면 그것을 우선 (검증용 파일 대신)
            for k, v in best["params"].items():
                setattr(args, k, v)
            if cli_aug:
                args.aug_file = cli_aug
            source = f"best_params.json (val F1 {best['f1']:.4f}, n_val {best['n_val']})"
        params = params_of(args)
        print(f"  사용 파라미터 [{source}]: {params}")
        test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
        decoder, _ = build(train, args)
        preds = decode_all(decoder, list(test.input), args.workers)
        sub["output"] = sub.ID.map(dict(zip(test.ID, preds)))
        path = os.path.join(args.out_dir, "submission_noisy_channel.csv")
        sub.to_csv(path, index=False, encoding="utf-8-sig")
        with open(os.path.join(args.out_dir, "submission_noisy_channel_params.json"), "w", encoding="utf-8") as f:
            json.dump({"params": params, "source": source,
                       "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=1)
        print(f"  saved {path} (+ _params.json)")


if __name__ == "__main__":
    main()
