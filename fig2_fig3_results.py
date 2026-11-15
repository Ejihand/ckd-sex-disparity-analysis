"""Figures 2 and 3, styled to match Figure 1 and safe in greyscale."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 9,
    'axes.linewidth': 0.9,
    'axes.edgecolor': '#000000',
    'xtick.color': '#000000',
    'ytick.color': '#000000',
    'text.color': '#000000',
    'axes.labelcolor': '#000000',
})

WOMEN = dict(facecolor='#3f3f3f', edgecolor='black', linewidth=0.9)
MEN = dict(facecolor='#ffffff', edgecolor='black', linewidth=0.9, hatch='////')
os.makedirs('fig', exist_ok=True)

# ===================================================== Figure 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6),
                               gridspec_kw={'width_ratios': [1, 1.15]})


def zero_stub(ax, xi, bw):
    """A zero-height bar is invisible; mark its footprint so the reader sees it."""
    ax.plot([xi - bw / 2, xi + bw / 2], [0, 0], color='black', linewidth=2.6,
            solid_capstyle='butt', zorder=5)

# --- panel a: late-stage FNR by sex, Model A vs Model B
labels = ['Model A\n(unmitigated)', 'Model B\n(mitigated)']
w_vals = [33.3, 0.0]
m_vals = [0.0, 0.0]
x = np.arange(2)
bw = 0.34
ax1.bar(x - bw / 2, w_vals, bw, label='Women', **WOMEN)
ax1.bar(x + bw / 2, m_vals, bw, label='Men', **MEN)
for xi, v in zip(x - bw / 2, w_vals):
    ax1.text(xi, v + 1.2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9,
             fontweight='bold')
for xi, v in zip(x + bw / 2, m_vals):
    ax1.text(xi, v + 1.4, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    zero_stub(ax1, xi, bw)
zero_stub(ax1, x[1] - bw / 2, bw)          # women under Model B are also zero
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel('Late-stage (G4/G5) false-negative rate (%)')
ax1.set_ylim(0, 44)
ax1.set_title('(a)  Missed cases by sex', fontsize=9.5,
              fontweight='bold', loc='left', pad=10)
ax1.annotate("Fisher's exact\np = 0.011", xy=(0.03, 26), fontsize=8,
             ha='left', va='center')
ax1.annotate('p = 1.000', xy=(1.0, 8), fontsize=8, ha='center', va='center')
ax1.legend(frameon=False, fontsize=8.5, loc='upper center', ncol=2,
           bbox_to_anchor=(0.62, 1.0))
ax1.spines[['top', 'right']].set_visible(False)

# --- panel b: mitigation path, fairness gain against accuracy
stages = ['Model A\nunmitigated', '+ re-weighing\n(in-processing)',
          '+ threshold\n(Model B)']
fnr = [33.3, 13.3, 0.0]
acc = [74.4, 75.5, 75.5]
xb = np.arange(3)
ax2.bar(xb, fnr, 0.5, **WOMEN)
for xi, v in zip(xb, fnr):
    ax2.text(xi, v + 1.4, f'{v:.1f}%', ha='center', va='bottom', fontsize=9,
             fontweight='bold')
    if v == 0:
        zero_stub(ax2, xi, 0.5)
ax2.set_xticks(xb)
ax2.set_xticklabels(stages, fontsize=8.2)
ax2.set_ylabel('Female late-stage FNR (%)')
ax2.set_ylim(0, 44)
ax2.spines[['top']].set_visible(False)
ax2.set_title('(b)  Each mitigation layer, and its cost', fontsize=9.5,
              fontweight='bold', loc='left', pad=10)

ax2b = ax2.twinx()
ax2b.plot(xb, acc, marker='o', color='black', linewidth=1.4, markersize=5,
          markerfacecolor='white', markeredgewidth=1.2, linestyle='--')
for xi, v in zip(xb, acc):
    ax2b.text(xi, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
ax2b.set_ylabel('Overall accuracy (%)')
ax2b.set_ylim(60, 90)
ax2b.spines[['top']].set_visible(False)
ax2b.legend(handles=[plt.Line2D([], [], color='black', linestyle='--', marker='o',
                                markerfacecolor='white', markersize=5,
                                label='overall accuracy (right axis)')],
            frameon=False, fontsize=8.2, loc='upper right',
            bbox_to_anchor=(1.0, 0.99))

fig.tight_layout()
fig.savefig('fig/Figure2_disparity_and_mitigation.pdf', bbox_inches='tight')
fig.savefig('fig/Figure2_disparity_and_mitigation.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ===================================================== Figure 3
fig, ax = plt.subplots(figsize=(5.6, 3.4))
groups = ['Women\n(n = 274)', 'Men\n(n = 219)']
observed = [0.055, 0.087]
pred_a = [0.040, 0.093]
pred_b = [0.050, 0.101]
x = np.arange(2)
bw = 0.26
ax.bar(x - bw, observed, bw, label='Observed prevalence',
       facecolor='#000000', edgecolor='black', linewidth=0.9)
ax.bar(x, pred_a, bw, label='Model A mean predicted',
       facecolor='#ffffff', edgecolor='black', linewidth=0.9, hatch='////')
ax.bar(x + bw, pred_b, bw, label='Model B mean predicted',
       facecolor='#b0b0b0', edgecolor='black', linewidth=0.9)
for xi, vals in zip(x, [[observed[0], pred_a[0], pred_b[0]],
                        [observed[1], pred_a[1], pred_b[1]]]):
    for off, v in zip([-bw, 0, bw], vals):
        ax.text(xi + off, v + 0.0025, f'{v:.3f}', ha='center', va='bottom',
                fontsize=8, zorder=8,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.8))
# a reference line at each group's observed prevalence makes the direction
# of the error readable at a glance
for xi, obs in zip(x, observed):
    ax.hlines(obs, xi - bw * 1.75, xi + bw * 1.75, colors='black',
              linestyles=(0, (4, 3)), linewidth=1.1, zorder=6)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel('Advanced-stage (G4/G5) risk')
ax.set_ylim(0, 0.132)
ax.legend(frameon=False, fontsize=8.3, loc='upper left', ncol=1)
ax.spines[['top', 'right']].set_visible(False)
ax.text(0.5, -0.185, 'dashed line marks the observed prevalence in that group',
        transform=ax.transAxes, ha='center', fontsize=8, style='italic')
fig.tight_layout()
fig.savefig('fig/Figure3_calibration_by_sex.pdf', bbox_inches='tight')
fig.savefig('fig/Figure3_calibration_by_sex.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print('wrote Figure 2 and Figure 3')
