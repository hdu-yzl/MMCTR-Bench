import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path


# ============================================================================
# Global configuration
# ============================================================================

# ---------------------------------------------------------------------------
# Global font-size configuration (aligned with fig_eff.py / figs1,2,4)
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

MEAN_VALUE_FONT_SIZE = VALUE_LABEL_FONT_SIZE

OUT_DIR = Path("src/analysis/plot/fig_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_LABEL_FONT_SIZE,
    "ytick.labelsize": TICK_LABEL_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================================
# Category configurations
# ============================================================================

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


# ============================================================================
# Model colors
# ============================================================================

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


# ============================================================================
# Model-category mapping
# ============================================================================

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


# ============================================================================
# Display order
# ============================================================================

model_order = [
    "MMMLP", "Diff-MSIN", "DMF",
    "MAKE", "M3SRec", "EM3", "PSRQ", "QARM",
    "NAML", "MB", "LMF", "SimCEN", "MTFN",
    "PAMD", "GMMF", "MARN",
]

group_order = ["C1", "C2", "C3", "C4"]

rate_order = [
    "rate=0.1",
    "rate=0.3",
    "rate=0.5",
    "rate=0.7",
]

rate_labels = ["0.1", "0.3", "0.5", "0.7"]


# ============================================================================
# Experimental results
# ============================================================================

robustness_cum = {
    "Diff-MSIN": [1.14, 2.02, 3.06, 3.36],
    "DMF":       [1.35, 2.13, 2.96, 3.41],

    "MAKE":      [0.97, 1.31, 1.79, 2.28],
    "M3SRec":    [0.89, 1.40, 1.91, 2.04],
    "PSRQ":      [0.99, 1.59, 2.00, 2.37],
    "QARM":      [0.85, 1.18, 1.83, 1.96],
    "EM3":       [0.69, 1.29, 1.70, 2.15],

    "NAML":      [0.73, 0.89, 1.44, 1.82],
    "MB":        [0.71, 0.97, 1.36, 1.70],
    "LMF":       [0.44, 1.01, 1.46, 1.66],
    "SimCEN":    [0.53, 1.62, 1.68, 1.60],
    "MTFN":      [0.77, 0.79, 1.12, 1.48],

    "MMMLP":     [0.40, 1.50, 3.09, 4.33],

    "PAMD":      [0.14, 0.56, 1.06, 1.25],
    "GMMF":      [0.34, 0.58, 0.98, 1.20],
    "MARN":      [0.30, 0.55, 0.63, 0.75],
}


# ============================================================================
# Data preprocessing
# ============================================================================

df_cum = pd.DataFrame(
    robustness_cum,
    index=rate_order,
).T

df_cum = df_cum.loc[model_order]

df_cum["Category"] = [
    model_category[model]
    for model in df_cum.index
]

group_models = {
    category: [
        model
        for model in model_order
        if model_category[model] == category
    ]
    for category in group_order
}

group_mean = (
    df_cum.groupby("Category")[rate_order]
    .mean()
    .loc[group_order]
)


# ============================================================================
# Construct display rows
#
# Each category contains only its individual model rows.
# ============================================================================

display_rows = []

for category in group_order:
    for model in group_models[category]:
        display_rows.append({
            "category": category,
            "name": model,
            "kind": "model",
        })

for row_index, row in enumerate(display_rows):
    row["y"] = row_index


# ============================================================================
# Marker styles
# ============================================================================

RATE_MARKER = {
    "rate=0.1": "o",
    "rate=0.3": "s",
    "rate=0.5": "^",
    "rate=0.7": "D",
}


# ============================================================================
# Figure
# ============================================================================

fig, ax = plt.subplots(
    figsize=(12, 5.8),
)

n_rows = len(display_rows)

global_max = df_cum[rate_order].to_numpy().max()
x_max = np.ceil(global_max * 1.10 * 10) / 10


# ============================================================================
# Category backgrounds
# ============================================================================

start_row = 0

for category in group_order:
    category_rows = [
        row
        for row in display_rows
        if row["category"] == category
    ]

    end_row = start_row + len(category_rows) - 1

    # Entire category background
    ax.axhspan(
        start_row - 0.5,
        end_row + 0.5,
        color=CATEGORY_COLOR[category],
        alpha=0.08,
        linewidth=0,
        zorder=0,
    )

    # Category separator
    if category != group_order[-1]:
        ax.axhline(
            end_row + 0.5,
            color="gray",
            linewidth=0.8,
            alpha=0.5,
            zorder=1,
        )

    start_row = end_row + 1


# ============================================================================
# Model trajectories
# ============================================================================

for row in display_rows:
    category = row["category"]
    row_name = row["name"]
    y = row["y"]

    values = df_cum.loc[
        row_name,
        rate_order,
    ].to_numpy(dtype=float)

    line_color = MODEL_COLOR[row_name]
    line_width = 1.9
    marker_size = 60
    marker_edge_color = "#FFFFFF"
    marker_edge_width = 0.8
    zorder = 3

    # Horizontal trajectory connecting four mask-rate results
    ax.plot(
        values,
        [y] * len(values),
        color=line_color,
        linewidth=line_width,
        alpha=1.0,
        solid_capstyle="round",
        zorder=zorder,
    )

    # Four rate markers
    for rate, value in zip(rate_order, values):
        ax.scatter(
            value,
            y,
            s=marker_size,
            marker=RATE_MARKER[rate],
            facecolor=line_color,
            edgecolor=marker_edge_color,
            linewidth=marker_edge_width,
            zorder=zorder + 1,
        )


# ============================================================================
# Y-axis labels
# ============================================================================

ax.set_yticks([
    row["y"]
    for row in display_rows
])

ax.set_yticklabels([
    row["name"]
    for row in display_rows
])

for tick_label in ax.get_yticklabels():
    tick_label.set_color("#1F1F1F")
    tick_label.set_fontweight("normal")
    tick_label.set_fontsize(TICK_LABEL_FONT_SIZE)


# First row should appear at the top
ax.invert_yaxis()


# ============================================================================
# X-axis
# ============================================================================

ax.set_xlim(0, x_max)

ax.set_xticks(
    np.arange(
        0,
        np.floor(x_max) + 1,
        1,
    )
)

ax.set_xlabel(
    "Relative AUC drop (%)",
    fontsize=AXIS_LABEL_FONT_SIZE,
    labelpad=8,
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.25,
    zorder=0,
)


# ============================================================================
# Axis styling
# ============================================================================

# Full box around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.1)

ax.tick_params(
    axis="x",
    direction="out",
    length=4,
    width=1.0,
)

ax.tick_params(
    axis="y",
    length=0,
    pad=8,
)


# ============================================================================
# Bottom legend: category colors + mask-rate markers
# ============================================================================

category_handles = [
    Patch(
        facecolor=CATEGORY_COLOR[category],
        alpha=0.65,
        label=CATEGORY_NAME[category],
    )
    for category in group_order
]

rate_handles = [
    Line2D(
        [0],
        [0],
        marker=RATE_MARKER[rate],
        linestyle="None",
        markerfacecolor="#555555",
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=8.5,
        label=f"Mask rate = {label}",
    )
    for rate, label in zip(
        rate_order,
        rate_labels,
    )
]

legend_handles = category_handles + rate_handles

bottom_legend = fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=4,
    frameon=False,
    fontsize=LEGEND_FONT_SIZE,
    handletextpad=0.45,
    columnspacing=1.8,
)


# ============================================================================
# Layout
# ============================================================================

fig.subplots_adjust(
    left=0.19,
    right=0.96,
    top=0.97,
    bottom=0.13,
)


# ============================================================================
# Save
# ============================================================================

pdf_path = (
    OUT_DIR
    / "exp3_missing_modality_trajectory_plot.pdf"
)

png_path = (
    OUT_DIR
    / "exp3_missing_modality_trajectory_plot.png"
)

fig.savefig(
    pdf_path,
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    bbox_extra_artists=[bottom_legend],
)

fig.savefig(
    png_path,
    format="png",
    dpi=300,
    bbox_inches="tight",
    bbox_extra_artists=[bottom_legend],
)

plt.close(fig)

print(f"Saved PDF: {pdf_path}")
print(f"Saved PNG: {png_path}")