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

exp_data = {
    "KL": {
        "PAMD": 1.00, "MARN": 0.77, "GMMF": 0.44,
        "LMF": 0.00, "NAML": -0.03, "SimCEN": -0.03,
        "MTFN": -0.07, "MB": -0.09, "EM3": -0.12,
        "MAKE": -0.27, "M3SRec": -0.28,
        "PSRQ": -0.56, "QARM": 0.06,
        "MMMLP": -0.15, "Diff-MSIN": -0.24, "DMF": -0.51,
    },

    "InfoNCE": {
        "MARN": 0.73, "PAMD": 0.46, "GMMF": 0.10,
        "MB": 0.01, "NAML": -0.11, "SimCEN": -0.19,
        "MTFN": -0.21, "LMF": -0.24,
        "MAKE": -0.81, "EM3": -0.36, "M3SRec": -0.70,
        "PSRQ": -0.60, "QARM": -0.75,
        "MMMLP": -1.15, "Diff-MSIN": -2.48, "DMF": -2.85,
    },

    "Cosine": {
        "GMMF": 0.31, "MTFN": 0.18, "NAML": 0.14,
        "MARN": 0.03, "PAMD": 0.00, "MB": -0.06,
        "SimCEN": -0.10, "LMF": -0.16,
        "M3SRec": -0.35, "EM3": -0.51, "MAKE": -0.60,
        "PSRQ": -0.36, "QARM": -0.03,
        "MMMLP": -1.16, "DMF": -0.86, "Diff-MSIN": -2.19,
    },

    "MMD": {
        "PAMD": 3.12, "MARN": 1.24, "MTFN": 0.09,
        "GMMF": 0.07, "NAML": -0.02, "LMF": -0.07,
        "MB": -0.09, "SimCEN": -0.09,
        "MAKE": -0.17, "MMMLP": -0.17, "M3SRec": -0.20,
        "EM3": -0.52, "PSRQ": -0.54, "QARM": -0.36,
        "Diff-MSIN": -1.38, "DMF": -1.03,
    },

    "Adv": {
        "MARN": 0.28, "GMMF": -0.05, "MTFN": -0.14,
        "MB": -0.20, "NAML": -0.21, "PAMD": -0.28,
        "SimCEN": -0.35, "LMF": -0.36,
        "EM3": -0.57, "MAKE": -0.49, "M3SRec": -0.66,
        "PSRQ": -0.46, "QARM": -0.82,
        "MMMLP": -1.86, "DMF": -0.72, "Diff-MSIN": -2.72,
    },

    "AVG": {
        "PAMD": 0.86, "MARN": 0.61, "GMMF": 0.17,
        "NAML": -0.04, "MTFN": -0.03, "MB": -0.09,
        "SimCEN": -0.15, "LMF": -0.17,
        "EM3": -0.42, "MAKE": -0.47, "M3SRec": -0.44,
        "PSRQ": -0.50, "QARM": -0.38,
        "MMMLP": -0.89, "DMF": -1.19, "Diff-MSIN": -1.82,
    },
}


def add_category_spans(ax, ordered_models):
    ymin, ymax = ax.get_ylim()
    start = 0

    while start < len(ordered_models):
        c = model_category[ordered_models[start]]
        end = start

        while (
            end + 1 < len(ordered_models)
            and model_category[ordered_models[end + 1]] == c
        ):
            end += 1

        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color=CATEGORY_COLOR[c],
            alpha=0.08,
            zorder=0,
        )

        if end + 0.5 < len(ordered_models) - 0.5:
            ax.axvline(
                end + 0.5,
                color="gray",
                linewidth=0.8,
                alpha=0.5,
            )

        start = end + 1


def add_value_labels(ax, values):
    ymin, ymax = ax.get_ylim()
    offset = 0.018 * (ymax - ymin)

    for i, v in enumerate(values):
        if v >= 0:
            ax.text(
                i,
                v + offset,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_LABEL_FONT_SIZE,
            )
        else:
            ax.text(
                i,
                v - offset,
                f"{v:.2f}",
                ha="center",
                va="top",
                fontsize=VALUE_LABEL_FONT_SIZE,
            )


loss_order = [
    "KL",
    "InfoNCE",
    "Cosine",
    "MMD",
    "Adv",
    "AVG",
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(24, 9),
    constrained_layout=True,
)
axes = axes.ravel()

subplot_letters = ["a", "b", "c", "d", "e", "f"]

for idx, (ax, loss_name) in enumerate(zip(axes, loss_order)):
    values = [exp_data[loss_name][m] for m in model_order]
    colors = [MODEL_COLOR[m] for m in model_order]

    ax.bar(
        range(len(model_order)),
        values,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"({subplot_letters[idx]}) {loss_name}", fontsize=TITLE_FONT_SIZE)
    ax.set_ylabel("AUC relative change (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(
        model_order,
        rotation=38,
        ha="right",
        fontsize=TICK_LABEL_FONT_SIZE,
    )
    ax.grid(axis="y", alpha=0.25, zorder=1)

    if loss_name == "AVG":
        ax.set_ylim(-2.1, 1.1)
    else:
        ax.set_ylim(-3.2, 3.5)

    add_category_spans(ax, model_order)
    # add_value_labels(ax, values)


legend_handles = [
    Patch(
        facecolor=CATEGORY_COLOR["C1"],
        alpha=0.65,
        label="TMIE",
    ),
    Patch(
        facecolor=CATEGORY_COLOR["C2"],
        alpha=0.65,
        label="CSAQ",
    ),
    Patch(
        facecolor=CATEGORY_COLOR["C3"],
        alpha=0.65,
        label="GFFI",
    ),
    Patch(
        facecolor=CATEGORY_COLOR["C4"],
        alpha=0.65,
        label="RDRR",
    ),
]

bottom_legend = fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.08),
    frameon=False,
    fontsize=LEGEND_FONT_SIZE,
)

save_path = OUT_DIR / "exp1_alignment_losses_four_categories.pdf"
fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight",
            bbox_extra_artists=[bottom_legend])
plt.close(fig)

print(f"Saved figure: {save_path}")