"""README 그림을 만든다: python docs/figures/make_figures.py  →  docs/figures/stages.svg
점수는 같은 검증 분할(train 의 10%, 1,126문장)의 문자 단위 F1 (results/ 의 실행 기록)."""
from pathlib import Path

STAGES = [  # (이름, F1, 종류) 종류: base=자체, repo=참고 저장소 아이디어 적용, solo=단독/비교용, final=최종
    ("01 노이즈 채널 + 어절 사전", 0.9834, "base"),
    ("05 태거 v2 (단독)", 0.9758, "solo"),
    ("08 확실한 어절 고정 + LLM 문맥", 0.9894, "base"),
    ("09 고정 풀기 · 후보 확대 · LLM 미세조정", 0.9908, "base"),
    ("10 생성 문장으로 어휘 증강", 0.9912, "base"),
    ("11 태거 v3 (참고 저장소 증강)", 0.9915, "repo"),
    ("12b 외부 모델 없이 (글자 언어모델)", 0.9871, "solo"),
    ("13b KoCharELECTRA 채점기", 0.9922, "repo"),
    ("14 최종 결합", 0.9937, "final"),
]
COL = {"base": "#7b8794", "solo": "#c3cad3", "repo": "#d9822b", "final": "#1f5fa8"}
LO, HI = 0.970, 0.995
W, LW, BW, RH, TOP = 760, 300, 380, 30, 46


def main():
    h = TOP + RH * len(STAGES) + 52
    x = lambda v: LW + (v - LO) / (HI - LO) * BW
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
         'font-family="Pretendard, \'Noto Sans KR\', \'Malgun Gothic\', sans-serif">',
         f'<rect width="{W}" height="{h}" fill="#ffffff"/>',
         '<text x="20" y="28" font-size="16" font-weight="700" fill="#1d2733">단계별 검증 F1 (문자 단위)</text>']
    for t in (0.970, 0.975, 0.980, 0.985, 0.990, 0.995):
        o.append(f'<line x1="{x(t):.1f}" y1="{TOP-6}" x2="{x(t):.1f}" y2="{TOP+RH*len(STAGES)}" stroke="#e6e9ee"/>')
        o.append(f'<text x="{x(t):.1f}" y="{TOP+RH*len(STAGES)+16}" font-size="11" fill="#7b8794" text-anchor="middle">{t:.3f}</text>')
    for i, (n, v, k) in enumerate(STAGES):
        y = TOP + i * RH
        bold = ' font-weight="700"' if k == "final" else ""
        o.append(f'<text x="{LW-10}" y="{y+19}" font-size="13" fill="#1d2733" text-anchor="end"{bold}>{n}</text>')
        o.append(f'<rect x="{LW}" y="{y+6}" width="{x(v)-LW:.1f}" height="{RH-12}" rx="3" fill="{COL[k]}"/>')
        o.append(f'<text x="{x(v)+6:.1f}" y="{y+19}" font-size="12.5" fill="#1d2733"{bold}>{v:.4f}</text>')
    ly = TOP + RH * len(STAGES) + 38
    lx = 20
    for k, lab in (("base", "직접 만든 단계"), ("repo", "참고 저장소 아이디어 적용"), ("solo", "단독·비교용"), ("final", "최종")):
        o.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" rx="2" fill="{COL[k]}"/>'
                 f'<text x="{lx+18}" y="{ly}" font-size="12" fill="#4a5562">{lab}</text>')
        lx += 18 + 13 * len(lab) + 26
    o.append(f'<text x="{W-20}" y="{ly}" font-size="11" fill="#9aa3ad" text-anchor="end">막대는 0.970부터</text>')
    o.append("</svg>")
    Path(__file__).with_name("stages.svg").write_text("\n".join(o), encoding="utf-8")


if __name__ == "__main__":
    main()
