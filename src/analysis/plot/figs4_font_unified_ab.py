import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

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
    "MAKE": "#FDAE6B",
    "M3SRec": "#FD8D3C",
    "EM3": "#D94801",
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
    # C1
    "MMMLP": "C1",
    "Diff-MSIN": "C1",
    "DMF": "C1",

    # C2
    "MAKE": "C2",
    "M3SRec": "C2",
    "EM3": "C2",
    "PSRQ": "C2",
    "QARM": "C2",

    # C3
    "NAML": "C3",
    "MB": "C3",
    "LMF": "C3",
    "SimCEN": "C3",
    "MTFN": "C3",

    # C4
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

metric_order = [
    "CAT",
    "DMF",
    "DTA",
    "FQ-Former",
    "LMF",
    "MAF",
    "MTFN",
    "SimCEN",
    "SRC",
]

all_metric_order = metric_order + ["avg"]

data = {
    # C4
    "PAMD": [
        0.00, -0.57, -0.39, -0.74, -0.21,
        -0.66, -0.25, -1.86, -2.03, -0.75,
    ],
    "MARN": [
        -0.33, -0.38, -1.17, 0.13, -0.71,
        0.00, -0.23, -1.58, -0.07, -0.48,
    ],
    "GMMF": [
        1.32, -0.76, -0.93, 0.76, 1.00,
        0.85, 0.75, -1.16, -0.88, -0.11,
    ],

    # C3
    "LMF": [
        0.37, -1.69, -1.86, -0.19, 0.00,
        -0.09, -0.20, -2.09, -1.81, -0.84,
    ],
    "MTFN": [
        0.30, -1.76, -1.92, -0.26, -0.02,
        -0.17, 0, -0.36, -1.88, -0.67,
    ],
    "MB": [
        -0.07, -0.72, -0.74, -0.40, -0.19,
        -0.56, -0.26, -2.01, -1.49, -0.72,
    ],
    "SimCEN": [
        0.75, -1.32, -1.48, 0.19, 0.43,
        0.28, 0.18, 0.00, -1.44, -0.28,
    ],
    "NAML": [
        0.00, -0.25, 0.11, -1.16, -0.21,
        0.22, -0.24, -1.67, -0.84, -0.45,
    ],

    # C2
    "EM3": [
        2.26, 2.60, 0.88, 0.00, -0.23,
        0.98, 1.60, 0.53, -1.94, 0.74,
    ],
    "MAKE": [
        0.89, 0.55, 0.36, -0.03, 0.18,
        -0.28, 2.46, -2.88, -0.19, 0.17,
    ],
    "M3SRec": [
        -0.08, -0.59, -0.42, -0.37, 0.11,
        0.76, 0.29, 1.66, -0.55, 0.09,
    ],
    "PSRQ": [
        0.00, 0.83, -0.09, 0.57, 0.34,
        -0.05, 0.41, -1.12, -0.73, 0.02,
    ],
    "QARM": [
        0.00, 1.29, -0.11, -0.22, 0.14,
        -0.13, 0.08, 1.76, -1.61, 0.13,
    ],

    # C1
    "MMMLP": [
        0.07, -1.19, -1.02, -0.44, -0.38,
        -0.31, -3.14, -1.89, -1.94, -1.14,
    ],
    "Diff-MSIN": [
        0.72, -1.68, -1.39, -3.76, -0.51,
        -0.60, -3.47, -2.27, 0.00, -1.44,
    ],
    "DMF": [
        -0.62, 0.00, -0.57, -2.56, -1.00,
        -0.77, -0.26, -0.93, -5.72, -1.38,
    ],
}


df = pd.DataFrame.from_dict(
    data,
    orient="index",
    columns=all_metric_order,
)

df = df.loc[model_order]
df["Category"] = [model_category[m] for m in df.index]

# 趋势图使用所有模型在各 backbone 上的整体均值
df_overall_mean = df[metric_order].mean(axis=0)


def get_group_spans(ordered_models):
    spans = []
    start = 0

    while start < len(ordered_models):
        category = model_category[ordered_models[start]]
        end = start

        while (
            end + 1 < len(ordered_models)
            and model_category[ordered_models[end + 1]] == category
        ):
            end += 1

        spans.append((start, end, category))
        start = end + 1

    return spans


def add_category_background_barh(ax, ordered_models, alpha=0.08):
    spans = get_group_spans(ordered_models)

    for start, end, category in spans:
        ax.axhspan(
            start - 0.5,
            end + 0.5,
            color=CATEGORY_COLOR[category],
            alpha=alpha,
            zorder=0,
        )

        if end + 0.5 < len(ordered_models) - 0.5:
            ax.axhline(
                end + 0.5,
                color="gray",
                linewidth=0.9,
                alpha=0.5,
                zorder=1,
            )


def add_bar_labels(ax, values, y_positions):
    xmin, xmax = ax.get_xlim()
    offset = 0.015 * (xmax - xmin)

    for y, value in zip(y_positions, values):
        if value >= 0:
            ax.text(
                value + offset,
                y,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=VALUE_LABEL_FONT_SIZE,
            )
        else:
            ax.text(
                value - offset,
                y,
                f"{value:.2f}",
                va="center",
                ha="right",
                fontsize=VALUE_LABEL_FONT_SIZE,
            )


fig = plt.figure(figsize=(12, 5.8))

gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[1.0, 1.0],
    wspace=0.24,
)

ax_avg = fig.add_subplot(gs[0, 0])
ax_group = fig.add_subplot(gs[0, 1])


# ============================================================
# 1. AVG 横向条形图（原图 b -> 现图 a）
# ============================================================

avg_vals = df["avg"].values
y_positions = np.arange(len(model_order))

add_category_background_barh(
    ax_avg,
    model_order,
    alpha=0.08,
)

ax_avg.barh(
    y_positions,
    avg_vals,
    color=[MODEL_COLOR[m] for m in model_order],
    edgecolor="white",
    linewidth=0.8,
    zorder=2,
)

ax_avg.axvline(
    0,
    color="black",
    linewidth=0.9,
)

ax_avg.set_yticks(y_positions)
ax_avg.set_yticklabels(
    model_order,
    fontsize=TICK_LABEL_FONT_SIZE,
    color="black",
)

ax_avg.invert_yaxis()

ax_avg.set_xlabel(
    "Average relative change (%)",
    fontsize=AXIS_LABEL_FONT_SIZE,
)

ax_avg.set_title(
    "(a) Model-wise Average Change",
    fontsize=TITLE_FONT_SIZE,
)

ax_avg.grid(
    axis="x",
    alpha=0.25,
    linestyle="--",
    zorder=1,
)

x_left = min(
    avg_vals.min() - 0.65,
    -2.2,
)

x_right = max(
    avg_vals.max() + 0.65,
    1.2,
)

ax_avg.set_xlim(
    x_left,
    x_right,
)

# add_bar_labels(
#     ax_avg,
#     avg_vals,
#     y_positions,
# )


# ============================================================
# 2. 所有模型的整体均值趋势图（原图 c -> 现图 b）
# ============================================================

x = np.arange(len(metric_order))
y = df_overall_mean.values

ax_group.plot(
    x,
    y,
    marker="o",
    linewidth=2.6,
    markersize=6,
    color="black",
    zorder=3,
)

ax_group.axhline(
    0,
    color="black",
    linewidth=0.9,
)

y_range = max(y.max() - y.min(), 1.0)
label_offset = 0.045 * y_range

# 留出足够上下边距，避免数值标注超出边框
ax_group.set_ylim(
    y.min() - 4.0 * label_offset,
    y.max() + 4.0 * label_offset,
)

ax_group.set_xticks(x)
ax_group.set_xticklabels(
    metric_order,
    rotation=20,
    ha="right",
    fontsize=TICK_LABEL_FONT_SIZE,
    color="black",
)
ax_group.tick_params(axis="x", colors="black")

ax_group.set_ylabel(
    "Mean relative change (%)",
    fontsize=AXIS_LABEL_FONT_SIZE,
)

ax_group.set_title(
    "(b) Operator-wise Average Change",
    fontsize=TITLE_FONT_SIZE,
)

ax_group.grid(
    axis="y",
    alpha=0.25,
    linestyle="--",
    zorder=1,
)


fig.subplots_adjust(
    left=0.08,
    right=0.985,
    top=0.92,
    bottom=0.18,
)
legend_handles = [
    Patch(facecolor=CATEGORY_COLOR["C1"], alpha=0.65, label="TMIE"),
    Patch(facecolor=CATEGORY_COLOR["C2"], alpha=0.65, label="CSAQ"),
    Patch(facecolor=CATEGORY_COLOR["C3"], alpha=0.65, label="GFFI"),
    Patch(facecolor=CATEGORY_COLOR["C4"], alpha=0.65, label="RDRR"),
]

bottom_legend = fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.05),
    frameon=False,
    fontsize=LEGEND_FONT_SIZE,
)
save_path = (
    OUT_DIR
    / "exp4_cross_backbone_composite_overall_mean_trend_ab.pdf"
)

fig.savefig(
    save_path, format="pdf",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved figure: {save_path}")
