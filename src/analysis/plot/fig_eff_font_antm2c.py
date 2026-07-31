import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_LABEL_FONT_SIZE,
    "ytick.labelsize": TICK_LABEL_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})

# ---------------------------------------------------------------------------
# 颜色：完全保留原配色，并按类别分组用于图例
# ---------------------------------------------------------------------------
CATEGORIES = {
    "TMIE": ["MMMLP", "Diff-MSIN", "DMF"],
    "CSAQ": ["MAKE", "M3SRec", "EM3", "PSRQ", "QARM"],
    "GFFI": ["NAML", "MB", "LMF", "SimCEN", "MTFN"],
    "RDRR": ["PAMD", "GMMF", "MARN"],
}

MODEL_COLOR = {
    "MMMLP": "#9ECAE1", "Diff-MSIN": "#6BAED6", "DMF": "#2171B5",
    "MAKE": "#FDBE85", "M3SRec": "#FDAE6B", "EM3": "#FD8D3C",
    "PSRQ": "#E6550D", "QARM": "#A63603",
    "NAML": "#C7E9C0", "MB": "#A1D99B", "LMF": "#74C476",
    "SimCEN": "#31A354", "MTFN": "#006D2C",
    "PAMD": "#BCBDDC", "GMMF": "#9E9AC8", "MARN": "#6A51A3",
}

# 类别平均值星星的高对比配色（与各类别色系呼应但更突出）
CATEGORY_COLOR = {
    "TMIE": "#08306B",   # 深蓝
    "CSAQ": "#7F2704",   # 深橙/棕
    "GFFI": "#00441B",   # 深绿
    "RDRR": "#3F007D",   # 深紫
}

raw_data = {
    "(a)AntM2C": {
        "Diff-MSIN": {"train": 4624.34, "infer": 71.35, "param": "29,717,023"},
        "PSRQ": {"train": 176.61, "infer": 18.99, "param": "23,901,190"},
        "LMF": {"train": 141.51, "infer": 12.67, "param": "21,189,001"},
        "MAKE": {"train": 88.35, "infer": 11.73, "param": "24,198,791"},
        "SimCEN": {"train": 77.84, "infer": 11.36, "param": "22,087,051"},
        "GMMF": {"train": 123.21, "infer": 11.57, "param": "22,198,917"},
        "QARM": {"train": 393.13, "infer": 9.07, "param": "24,615,811"},
        "DMF": {"train": 203.16, "infer": 12.27, "param": "23,994,471"},
        "MARN": {"train": 127.55, "infer": 18.55, "param": "22,715,664"},
        "MTFN": {"train": 411.44, "infer": 29.57, "param": "23,310,339"},
        "NAML": {"train": 539.20, "infer": 12.59, "param": "20,306,944"},
        "EM3": {"train": 202.09, "infer": 15.89, "param": "24,067,975"},
        "MMMLP": {"train": 642.71, "infer": 17.87, "param": "24,045,211"},
        "PAMD": {"train": 679.29, "infer": 14.10, "param": "21,643,011"},
        "M3SRec": {"train": 149.87, "infer": 15.09, "param": "23,934,355"},
        "MB": {"train": 432.17, "infer": 17.09, "param": "22,017,155"},
    },
}


def parse_param(x):
    return float(str(x).replace(",", "").strip())


rows = []
for dataset, model_dict in raw_data.items():
    for model, vals in model_dict.items():
        param = parse_param(vals["param"])
        rows.append({
            "Dataset": dataset, "Model": model,
            "Train": float(vals["train"]), "Infer": float(vals["infer"]),
            "Param_M": param / 1e6,
        })

df = pd.DataFrame(rows)
datasets = ["(a)AntM2C"]

# ---------------------------------------------------------------------------
# 画布：1 行 x 2 列 (train / infer) 并排
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2, figsize=(12, 5),
    constrained_layout=True,
)
fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.04, wspace=0.06, hspace=0.05)


def plot_panel(ax, sub, metric, marker):
    for _, row in sub.iterrows():
        ax.scatter(
            row["Param_M"], row[metric],
            s=110, marker=marker,
            color=MODEL_COLOR[row["Model"]],
            edgecolor="black", linewidth=0.5,
            zorder=3, alpha=0.95,
        )
    # 每个类别的平均值：用大星星标记
    for cat, models in CATEGORIES.items():
        cat_sub = sub[sub["Model"].isin(models)]
        if cat_sub.empty:
            continue
        mean_x = cat_sub["Param_M"].mean()
        mean_y = cat_sub[metric].mean()
        ax.scatter(
            mean_x, mean_y,
            s=430, marker="*",
            color=CATEGORY_COLOR[cat],
            edgecolor="white", linewidth=1.0,
            zorder=6, alpha=1.0,
        )
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", alpha=0.30)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)


dataset = datasets[0]
sub = df[df["Dataset"] == dataset].copy()

plot_panel(axes[0], sub, "Train", marker="s")
plot_panel(axes[1], sub, "Infer", marker="o")

# ---------------------------------------------------------------------------
# 纵轴：两幅图各自设定，但保持相同的对数跨度（≈2.08 decade）与主刻度数量，
# 使左右刻度在视觉上对齐（每格代表相同倍数），同时数据都能舒展开
# ---------------------------------------------------------------------------
axes[0].set_ylim(50, 6000)
axes[0].yaxis.set_major_locator(FixedLocator([100, 1000]))
axes[1].set_ylim(5, 600)
axes[1].yaxis.set_major_locator(FixedLocator([10, 100]))

for ax in axes:
    ax.set_xlabel("Model size (M parameters)", fontsize=AXIS_LABEL_FONT_SIZE)

axes[0].set_ylabel("Training time (ms)", fontsize=AXIS_LABEL_FONT_SIZE)
axes[1].set_ylabel("Inference time (ms)", fontsize=AXIS_LABEL_FONT_SIZE)

# ---------------------------------------------------------------------------
# 图例 1：类别平均星星（★ = 类别平均值）—— 图下方，两端对齐
# ---------------------------------------------------------------------------
star_handles = [
    Line2D([0], [0], marker="*", color="w", label=f"{cat} mean",
           markerfacecolor=CATEGORY_COLOR[cat], markeredgecolor="white",
           markersize=20)
    for cat in CATEGORIES
]
shape_legend = fig.legend(
    handles=star_handles,
    loc="lower left", bbox_to_anchor=(0.03, -0.10, 0.94, 0.05),
    ncol=4, mode="expand",
    frameon=False, fontsize=LEGEND_FONT_SIZE,
)

# ---------------------------------------------------------------------------
# 图例 2：模型颜色 —— 图下方，均匀分布、两端对齐
# ---------------------------------------------------------------------------
color_handles = []
for cat, models in CATEGORIES.items():
    for m in models:
        color_handles.append(
            Patch(facecolor=MODEL_COLOR[m], edgecolor="black",
                  linewidth=0.4, label=m)
        )

color_legend = fig.legend(
    handles=color_handles,
    loc="lower left", bbox_to_anchor=(0.03, -0.23, 0.94, 0.05),
    ncol=8, mode="expand",
    frameon=False, fontsize=LEGEND_FONT_SIZE,
    handlelength=1.4, handletextpad=0.6,
)

save_path = OUT_DIR / "efficiency_antm2c.pdf"
fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight",
            bbox_extra_artists=[color_legend, shape_legend])
plt.close(fig)
print(f"Saved figure: {save_path}")
