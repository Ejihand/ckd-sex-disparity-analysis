"""Figure 1: cohort flow and study design, STROBE-style.

Boxes size themselves to their content so nothing overflows.
"""
import os

FONT = "Helvetica, Arial, sans-serif"
LH = 20          # line height
PAD = 13         # vertical padding inside a box
W = 1000

parts = []


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def box_h(lines):
    return len(lines) * LH + PAD * 2


def box(x, y, w, lines, bold_first=True, dashed=False, lw=1.6):
    h = box_h(lines)
    dash = ' stroke-dasharray="6 4"' if dashed else ''
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" '
                 f'fill="#ffffff" stroke="#000" stroke-width="{lw}"{dash}/>')
    ty = y + PAD + 15
    for i, ln in enumerate(lines):
        bold = (i == 0 and bold_first)
        parts.append(
            f'<text x="{x + w/2}" y="{ty + i*LH}" font-family="{FONT}" '
            f'font-size="{15 if bold else 14}" font-weight="{"600" if bold else "400"}" '
            f'fill="#000" text-anchor="middle">{esc(ln)}</text>')
    return y + h


def vline(x, y1, y2, head=True):
    mk = ' marker-end="url(#a)"' if head else ''
    parts.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#000" '
                 f'stroke-width="1.6"{mk}/>')


def hline(x1, x2, y, head=True):
    mk = ' marker-end="url(#a)"' if head else ''
    parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#000" '
                 f'stroke-width="1.6"{mk}/>')


def stage(x, y, text):
    parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="14" '
                 f'font-weight="700" fill="#000" text-anchor="start">{esc(text)}</text>')


def note(x, y, text):
    parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="12.5" '
                 f'font-style="italic" fill="#000" text-anchor="middle">{esc(text)}</text>')


MX, MW = 250, 420          # main column
CX = MX + MW / 2           # 460
EX, EW = 690, 296          # exclusion column
LBL = 22                   # stage label column
GAP = 58                   # vertical gap carrying a side branch
STEP = 28                  # plain vertical gap

body = []
y = 26

# ------------------------------------------------------------- enrolment
L = ['Routine serum-biochemistry records',
     'OAU Teaching Hospital and affiliated',
     'diagnostic centres, Osun State, Nigeria',
     'n = 2,510']
stage(LBL, y + box_h(L) / 2, 'Enrolment')
y = box(MX, y, MW, L)

EXC1 = ['Excluded: multivariate outlier screening',
        '(Isolation Forest, contamination 0.015)',
        'n = 38 (24 men, 14 women)']
mid = y + GAP / 2
vline(CX, y, y + GAP)
hline(CX, EX - 4, mid)
box(EX, mid - box_h(EXC1) / 2, EW, EXC1, bold_first=False, dashed=True)
y += GAP

L = ['n = 2,472 after outlier screening (98.49% retained)']
y = box(MX, y, MW, L, bold_first=False)

EXC2 = ['Excluded: paediatric (age < 18 years,',
        'MDRD not validated below 18 years)',
        'n = 9']
mid = y + GAP / 2
vline(CX, y, y + GAP)
hline(CX, EX - 4, mid)
box(EX, mid - box_h(EXC2) / 2, EW, EXC2, bold_first=False, dashed=True)
y += GAP

# ---------------------------------------------------------------- cohort
L = ['Analytic cohort   n = 2,463',
     '1,255 women  /  1,208 men',
     'mean age 49.1 ± 15.3 years (range 18–98)',
     '170 records (6.9%) KDIGO G4–G5']
stage(LBL, y + box_h(L) / 2, 'Cohort')
y = box(MX, y, MW, L, lw=2.3)

vline(CX, y, y + STEP)
y += STEP
L = ['KDIGO stage derived per record',
     'race-free four-variable MDRD',
     'G1 ≥ 90 · G2 60–89 · G3 30–59 · G4 15–29 '
     '· G5 < 15']
stage(LBL, y + box_h(L) / 2, 'Labelling')
y = box(MX, y, MW, L, bold_first=False)

vline(CX, y, y + STEP)
y += STEP
L = ['Stratified 80:20 split on KDIGO stage (seed 42)']
y = box(MX, y, MW, L, bold_first=False)

# --------------------------------------------------------- train / test
TW = 196
TLX, TRX = 252, 512
vline(CX, y, y + 16, head=False)
hline(TLX + TW / 2, TRX + TW / 2, y + 16, head=False)
vline(TLX + TW / 2, y + 16, y + STEP)
vline(TRX + TW / 2, y + 16, y + STEP)
y += STEP
L1 = ['Training set', 'n = 1,970']
box(TLX, y, TW, L1)
box(TRX, y, TW, L1 and ['Held-out test set', 'n = 493'])
y += box_h(L1)

# ---------------------------------------------------------------- models
MODW = 340
MAX_, MBX = 92, 500
note(CX, y + 22, 'biological sex withheld from the feature matrix of both arms')
vline(TLX + TW / 2, y, y + 32, head=False)
hline(MAX_ + MODW / 2, MBX + MODW / 2, y + 32, head=False)
vline(MAX_ + MODW / 2, y + 32, y + 44)
vline(MBX + MODW / 2, y + 32, y + 44)
y += 44

A = ['Model A  (unmitigated)',
     'XGBoost, sex-blind',
     'six features: age + five serum analytes',
     'unweighted loss, threshold 0.50']
B = ['Model B  (mitigated)',
     'identical features, algorithm, split, seed',
     '+ stage-by-sex re-weighing',
     '+ group-conditional threshold (τ ≈ 0.02)']
stage(LBL, y + box_h(A) / 2, 'Models')
box(MAX_, y, MODW, A)
box(MBX, y, MODW, B)
y += max(box_h(A), box_h(B))

# ------------------------------------------------------------- evaluation
vline(MAX_ + MODW / 2, y, y + 22, head=False)
vline(MBX + MODW / 2, y, y + 22, head=False)
hline(MAX_ + MODW / 2, MBX + MODW / 2, y + 22, head=False)
vline(CX, y + 22, y + STEP + 14)
y += STEP + 14

L = ['Evaluation on the held-out test set',
     'late-stage (G4/G5) false-negative rate by sex',
     'equalized-odds difference · disparate impact ratio',
     'calibration by sex (Brier score, ECE, CiL)',
     'Fisher’s exact test · 2,000-resample bootstrap']
stage(LBL, y + box_h(L) / 2, 'Audit')
y = box(MX, y, MW, L, lw=2.3)

vline(CX, y, y + STEP)
y += STEP
L = ['Robustness checks',
     '80 independent stratified re-partitions',
     'second model family on the study cohort: multilayer perceptron',
     'external cohort NHANES 2017–2018 (n = 5,071):',
     'XGBoost, multilayer perceptron, multinomial logistic regression']
y = box(MX, y, MW, L)

H = int(y + 26)

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}">',
       '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
       'markerHeight="7" orient="auto-start-reverse">'
       '<path d="M 0 0 L 10 5 L 0 10 z" fill="#000"/></marker></defs>',
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>'] + parts + ['</svg>']

os.makedirs('fig', exist_ok=True)
open('fig/Figure1_cohort_flow.svg', 'w', encoding='utf8').write('\n'.join(svg))
print(f'wrote fig/Figure1_cohort_flow.svg  ({W} x {H})')
