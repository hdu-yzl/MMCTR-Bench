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
    "KL Loss": {
        "PAMD": 1.00, "MARN": 0.77, "GMMF": 0.44,
        "LMF": 0.00, "NAML": -0.03, "SimCEN": -0.03,
        "MTFN": -0.07, "MB": -0.09, "EM3": -0.12,
        "MAKE": -0.27, "M3SRec": -0.28,
        "PSRQ": -0.56, "QARM": 0.06,
        "MMMLP": -0.15, "Diff-MSIN": -0.24, "DMF": -0.51,
    },

    "InfoNCE Loss": {
        "MARN": 0.73, "PAMD": 0.46, "GMMF": 0.10,
        "MB": 0.01, "NAML": -0.11, "SimCEN": -0.19,
        "MTFN": -0.21, "LMF": -0.24,
        "MAKE": -0.81, "EM3": -0.36, "M3SRec": -0.70,
        "PSRQ": -0.60, "QARM": -0.75,
        "MMMLP": -1.15, "Diff-MSIN": -2.48, "DMF": -2.85,
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


# Only keep panel (c) AVG — single subplot, no title, legend kept.
loss_name = "AVG"
values = [exp_data[loss_name][m] for m in model_order]
colors = [MODEL_COLOR[m] for m in model_order]

fig, ax = plt.subplots(figsize=(12, 5.8))

ax.bar(
    range(len(model_order)),
    values,
    color=colors,
    edgecolor="white",
    linewidth=0.8,
    zorder=2,
)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("AUC relative change (%)", fontsize=AXIS_LABEL_FONT_SIZE)
ax.set_xticks(range(len(model_order)))
ax.set_xticklabels(
    model_order,
    rotation=38,
    ha="right",
    fontsize=TICK_LABEL_FONT_SIZE,
)
ax.grid(axis="y", alpha=0.25, zorder=1)
ax.set_ylim(-2.1, 1.1)

add_category_spans(ax, model_order)


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
    bbox_to_anchor=(0.5, -0.12),
    frameon=False,
    fontsize=LEGEND_FONT_SIZE,
)

save_path = OUT_DIR / "exp1_alignment_losses_c_only.pdf"
fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight",
            bbox_extra_artists=[bottom_legend])
plt.close(fig)

print(f"Saved figure: {save_path}")
