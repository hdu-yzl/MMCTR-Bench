import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

exp2_data = {
    "few-shot": {
        "GMMF": 17.16,
        "MARN": 16.08,
        "PAMD": 14.92,

        "NAML": 13.76,
        "LMF": 13.45,
        "MB": 13.35,
        "SimCEN": 12.66,
        "MTFN": 12.83,

        "DMF": 11.69,
        "MMMLP": 11.94,
        "Diff-MSIN": 11.38,

        "EM3": 10.44,
        "M3SRec": 11.21,
        "MAKE": 10.11,
        "PSRQ": 11.35,
        "QARM": 10.79,
    },
    "zero-shot": {
        "GMMF": 17.42,
        "MARN": 16.28,
        "PAMD": 16.19,

        "LMF": 15.68,
        "SimCEN": 14.86,
        "NAML": 14.06,
        "MB": 13.64,
        "MTFN": 14.77,

        "Diff-MSIN": 12.52,
        "DMF": 13.17,
        "MMMLP": 12.13,

        "EM3": 11.33,
        "M3SRec": 11.45,
        "MAKE": 10.37,
        "PSRQ": 11.68,
        "QARM": 11.34,
    },
}


def add_category_spans(ax, ordered_models):
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
            alpha=0.08,
            zorder=0,
        )

        if end + 0.5 < len(ordered_models) - 0.5:
            ax.axvline(end + 0.5, color="gray", linewidth=0.8, alpha=0.5)

        start = end + 1


def add_value_labels(ax, values):
    ymin, ymax = ax.get_ylim()
    offset = 0.012 * (ymax - ymin)

    for i, v in enumerate(values):
        ax.text(
            i,
            v + offset,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_FONT_SIZE,
        )


# --- Merge few-shot and zero-shot into a single grouped bar chart ---
# Same per-model colors, distinguished by hatch pattern.
import numpy as np

x = np.arange(len(model_order))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(12, 5.8))

# few-shot: solid fill
few_values = [exp2_data["few-shot"][m] for m in model_order]
few_colors = [MODEL_COLOR[m] for m in model_order]
ax.bar(
    x - bar_width / 2,
    few_values,
    bar_width,
    color=few_colors,
    edgecolor="white",
    linewidth=0.8,
    zorder=2,
)

# zero-shot: same colors + hatch pattern
zero_values = [exp2_data["zero-shot"][m] for m in model_order]
zero_colors = [MODEL_COLOR[m] for m in model_order]
ax.bar(
    x + bar_width / 2,
    zero_values,
    bar_width,
    color=zero_colors,
    edgecolor="white",
    linewidth=0.8,
    hatch="////",
    zorder=2,
)

ax.set_ylabel("AUC relative drop (%)", fontsize=AXIS_LABEL_FONT_SIZE)
ax.set_xticks(x)
ax.set_xticklabels(model_order, rotation=38, ha="right", fontsize=TICK_LABEL_FONT_SIZE)
ax.set_ylim(9.0, 18.5)
ax.grid(axis="y", alpha=0.25, zorder=1)

# Category background spans still work (each model occupies one x unit)
add_category_spans(ax, model_order)

# --- Legend: categories + solid/hatched distinction ---
category_handles = [
    Patch(facecolor=CATEGORY_COLOR["C1"], alpha=0.65, label="TMIE"),
    Patch(facecolor=CATEGORY_COLOR["C2"], alpha=0.65, label="CSAQ"),
    Patch(facecolor=CATEGORY_COLOR["C3"], alpha=0.65, label="GFFI"),
    Patch(facecolor=CATEGORY_COLOR["C4"], alpha=0.65, label="RDRR"),
]
setting_handles = [
    Patch(facecolor="gray", label="few-shot"),
    Patch(facecolor="gray", hatch="////", label="zero-shot"),
]
legend_handles = category_handles + setting_handles

bottom_legend = fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.16),
    frameon=False,
    fontsize=LEGEND_FONT_SIZE,
)

save_path = OUT_DIR / "exp2_antm2c_cold_start_grouped.pdf"
fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight",
            bbox_extra_artists=[bottom_legend])
plt.close(fig)

print(f"Saved figure: {save_path}")
