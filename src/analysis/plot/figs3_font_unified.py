import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path


# ---------------------------------------------------------------------------
# Global font-size configuration (aligned with fig_eff.py)
# ---------------------------------------------------------------------------
FONT_SIZE = 18
TITLE_FONT_SIZE = 21
AXIS_LABEL_FONT_SIZE = 19
TICK_LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 17
CATEGORY_LABEL_FONT_SIZE = 15
VALUE_LABEL_FONT_SIZE = 12
SMALL_ANNOTATION_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 14


OUT_DIR = Path("src/analysis/plot/fig_final")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_LABEL_FONT_SIZE,
    "ytick.labelsize": TICK_LABEL_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})


CATEGORY_NAME = {
    "C1": "TMIE",
    "C2": "CSAQ",
    "C3": "GFFI",
    "C4": "RDRR",
}

CATEGORY_COLOR = {
    "C1": "#3182BD",
    "C2": "#E6550D",
    "C3": "#31A354",
    "C4": "#756BB1",
}

MODEL_COLOR = {
    # C1
    "MMMLP": "#9ECAE1",
    "Diff-MSIN": "#6BAED6",
    "DMF": "#2171B5",

    # C2
    "MAKE": "#FDBE85",
    "M3SRec": "#FDAE6B",
    "EM3": "#FD8D3C",
    "PSRQ": "#E6550D",
    "QARM": "#A63603",

    # C3
    "NAML": "#C7E9C0",
    "MB": "#A1D99B",
    "LMF": "#74C476",
    "SimCEN": "#31A354",
    "MTFN": "#006D2C",

    # C4
    "PAMD": "#BCBDDC",
    "GMMF": "#9E9AC8",
    "MARN": "#6A51A3",
}

model_category = {
    "MMMLP": "C1",
    "Diff-MSIN": "C1",
    "DMF": "C1",

    "MAKE": "C2",
    "M3SRec": "C2",
    "EM3": "C2",
    "PSRQ": "C2",
    "QARM": "C2",

    "NAML": "C3",
    "MB": "C3",
    "LMF": "C3",
    "SimCEN": "C3",
    "MTFN": "C3",

    "PAMD": "C4",
    "GMMF": "C4",
    "MARN": "C4",
}

model_order = [
    "MMMLP", "Diff-MSIN", "DMF",
    "MAKE", "M3SRec", "EM3", "PSRQ", "QARM",
    "NAML", "MB", "LMF", "SimCEN", "MTFN",
    "PAMD", "GMMF", "MARN",
]

rate_order = ["rate=0.1", "rate=0.3", "rate=0.5", "rate=0.7"]

robustness_cum = {
    "Diff-MSIN": [1.14, 2.02, 3.06, 3.36],
    "DMF":       [1.35, 2.13, 2.96, 3.41],
    "MAKE":      [0.97, 1.31, 1.79, 2.28],
    "M3SRec":    [0.89, 1.40, 1.91, 2.04],
    "PSRQ":      [0.99, 1.59, 2.00, 2.37],
    "QARM":      [0.85, 1.18, 1.83, 1.96],
    "MTFN":      [0.77, 0.79, 1.12, 1.48],
    "NAML":      [0.73, 0.89, 1.44, 1.82],
    "MB":        [0.71, 0.97, 1.36, 1.70],
    "SimCEN":    [0.53, 1.62, 1.68, 1.60],
    "EM3":       [0.69, 1.29, 1.70, 2.15],
    "LMF":       [0.44, 1.01, 1.46, 1.66],
    "MMMLP":     [0.40, 1.50, 3.09, 4.33],
    "GMMF":      [0.34, 0.58, 0.98, 1.20],
    "MARN":      [0.30, 0.55, 0.63, 0.75],
    "PAMD":      [0.14, 0.56, 1.06, 1.25],
}

df_cum = pd.DataFrame(robustness_cum, index=rate_order).T
df_cum = df_cum.loc[model_order]
df_cum["Category"] = [model_category[m] for m in df_cum.index]

df_stack = df_cum[rate_order].copy()
df_stack["rate=0.7"] = df_cum["rate=0.7"] - df_cum["rate=0.5"]
df_stack["rate=0.5"] = df_cum["rate=0.5"] - df_cum["rate=0.3"]
df_stack["rate=0.3"] = df_cum["rate=0.3"] - df_cum["rate=0.1"]
df_stack["rate=0.1"] = df_cum["rate=0.1"]
df_stack_plot = df_stack.clip(lower=0)


def add_category_spans(ax, ordered_models, y_text_ratio=0.97, alpha=0.07):
    ymin, ymax = ax.get_ylim()
    start = 0

    while start < len(ordered_models):
        c = model_category[ordered_models[start]]
        end = start

        while end + 1 < len(ordered_models) and model_category[ordered_models[end + 1]] == c:
            end += 1

        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color=CATEGORY_COLOR[c],
            alpha=alpha,
            zorder=0,
        )

        if end + 0.5 < len(ordered_models) - 0.5:
            ax.axvline(end + 0.5, color="gray", linewidth=0.8, alpha=0.45)

        start = end + 1


def smooth_line(x, y, num=160):
    x = np.asarray(x)
    y = np.asarray(y)
    x_new = np.linspace(x.min(), x.max(), num)

    try:
        from scipy.interpolate import make_interp_spline
        spline = make_interp_spline(x, y, k=min(3, len(x) - 1))
        y_new = spline(x_new)
        y_new = np.maximum(y_new, 0)
    except Exception:
        y_new = np.interp(x_new, x, y)

    return x_new, y_new


def add_total_labels(ax, x, totals):
    ymin, ymax = ax.get_ylim()
    offset = 0.018 * (ymax - ymin)

    for xi, v in zip(x, totals):
        ax.text(
            xi,
            v + offset,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_FONT_SIZE,
        )


def add_axes_frame(ax, lw=1.3):
    for side in ["top", "bottom", "left", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(lw)
        ax.spines[side].set_color("black")


group_order = ["C1", "C2", "C3", "C4"]
group_mean = (
    df_cum.groupby("Category")[rate_order]
    .mean()
    .loc[group_order]
)

# =========================
# 左右排列：左边 Model-level，右边 Group-level
# =========================
fig, (ax_left, ax_right) = plt.subplots(
    1, 2,
    figsize=(24, 6),
    gridspec_kw={"width_ratios": [2.2, 1.0]},
)

# =========================
# 左图：Model-level stacked bar
# =========================
x_model = np.arange(len(model_order))
bottom = np.zeros(len(model_order))

hatch_map = {
    "rate=0.1": "///",
    "rate=0.3": "\\\\\\",
    "rate=0.5": "xxx",
    "rate=0.7": "...",
}

for r in rate_order:
    vals = df_stack_plot.loc[model_order, r].values
    colors = [MODEL_COLOR[m] for m in model_order]

    ax_left.bar(
        x_model,
        vals,
        bottom=bottom,
        width=0.68,
        color=colors,
        edgecolor="black",
        linewidth=0.75,
        hatch=hatch_map[r],
        zorder=2,
        alpha=0.95,
    )

    bottom += vals

total_vals = df_cum.loc[model_order, "rate=0.7"].values

ax_left.set_ylim(0, max(total_vals) * 1.22)
add_category_spans(ax_left, model_order, y_text_ratio=0.965, alpha=0.065)
# add_total_labels(ax_left, x_model, total_vals)

ax_left.set_xticks(x_model)
ax_left.set_xticklabels(model_order, rotation=32, ha="right", fontsize=TICK_LABEL_FONT_SIZE)
ax_left.set_ylabel("Relative drop (%)", fontsize=AXIS_LABEL_FONT_SIZE)
ax_left.set_title("(a) Model-level stacked degradation", fontsize=TITLE_FONT_SIZE, pad=8)
ax_left.grid(axis="y", alpha=0.22, linestyle="--", zorder=1)
add_axes_frame(ax_left)

# =========================
# 右图：Group-level mean
# =========================
x_rate = np.arange(len(rate_order))

for c in group_order:
    y = group_mean.loc[c, rate_order].values
    x_smooth, y_smooth = smooth_line(x_rate, y)

    ax_right.fill_between(
        x_smooth,
        y_smooth,
        0,
        color=CATEGORY_COLOR[c],
        alpha=0.12,
        zorder=1,
    )

    ax_right.plot(
        x_smooth,
        y_smooth,
        color=CATEGORY_COLOR[c],
        linewidth=2.8,
        zorder=3,
    )

    ax_right.scatter(
        x_rate,
        y,
        color=CATEGORY_COLOR[c],
        edgecolor="white",
        linewidth=1.0,
        s=58,
        zorder=4,
    )

    # for i, v in enumerate(y):
    #     ax_right.text(
    #         i,
    #         v + 0.04,
    #         f"{v:.2f}",
    #         ha="center",
    #         va="bottom",
    #         fontsize=ANNOTATION_FONT_SIZE,
    #         color=CATEGORY_COLOR[c],
    #     )

ax_right.set_xticks(x_rate)
ax_right.set_xticklabels(["0.1", "0.3", "0.5", "0.7"], fontsize=TICK_LABEL_FONT_SIZE)
ax_right.set_xlabel("Mask rate", fontsize=AXIS_LABEL_FONT_SIZE)
ax_right.set_ylabel("Group mean drop (%)", fontsize=AXIS_LABEL_FONT_SIZE)
ax_right.set_title("(b) Group-level mean degradation", fontsize=TITLE_FONT_SIZE, pad=8)
ax_right.set_ylim(0, group_mean.values.max() * 1.22)
ax_right.grid(axis="y", alpha=0.22, linestyle="--")
add_axes_frame(ax_right)

# =========================
# Legends
# =========================
group_legend = [
    Line2D(
        [0], [0],
        color=CATEGORY_COLOR["C1"],
        marker="o",
        lw=2.5,
        label="TMIE",
    ),
    Line2D(
        [0], [0],
        color=CATEGORY_COLOR["C2"],
        marker="o",
        lw=2.5,
        label="CSAQ",
    ),
    Line2D(
        [0], [0],
        color=CATEGORY_COLOR["C3"],
        marker="o",
        lw=2.5,
        label="GFFI",
    ),
    Line2D(
        [0], [0],
        color=CATEGORY_COLOR["C4"],
        marker="o",
        lw=2.5,
        label="RDRR",
    ),
]

rate_legend = [
    Patch(facecolor="white", edgecolor="black", hatch="///", label="rate=0.1"),
    Patch(facecolor="white", edgecolor="black", hatch="\\\\\\", label="rate=0.3 - rate=0.1"),
    Patch(facecolor="white", edgecolor="black", hatch="xxx", label="rate=0.5 - rate=0.3"),
    Patch(facecolor="white", edgecolor="black", hatch="...", label="rate=0.7 - rate=0.5"),
]

bottom_legend = fig.legend(
    handles=group_legend + rate_legend,
    loc="lower left",
    ncol=4,
    frameon=False,
    fontsize=LEGEND_FONT_SIZE,
    mode="expand",
    bbox_to_anchor=(0.04, -0.06, 0.92, 0.05),
)

fig.subplots_adjust(
    left=0.055,
    right=0.99,
    top=0.93,
    bottom=0.22,
    wspace=0.055,
)

png_path = OUT_DIR / "exp3_robustness_four_categories_side_by_side.pdf"
fig.savefig(png_path, format="pdf", dpi=300, bbox_inches="tight",
            bbox_extra_artists=[bottom_legend])
plt.close(fig)

print(f"Saved PNG: {png_path}")