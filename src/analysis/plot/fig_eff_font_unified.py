import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

# ---------------------------------------------------------------------------
# Global font-size configuration (aligned with fig_eff.py)
# ---------------------------------------------------------------------------
FONT_SIZE = 15
TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 13
LEGEND_FONT_SIZE = 14
CATEGORY_LABEL_FONT_SIZE = 12
VALUE_LABEL_FONT_SIZE = 9
SMALL_ANNOTATION_FONT_SIZE = 10
ANNOTATION_FONT_SIZE = 11

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
    "(a) AntM²C": {
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
    "(b) MicroLens": {
        "Diff-MSIN": {"train": 2933.04, "infer": 356.52, "param": "145,649,174"},
        "PSRQ": {"train": 127.13, "infer": 21.59, "param": "144,026,886"},
        "LMF": {"train": 120.29, "infer": 11.74, "param": "141,469,321"},
        "MAKE": {"train": 76.62, "infer": 9.16, "param": "143,821,703"},
        "SimCEN": {"train": 71.73, "infer": 8.77, "param": "140,609,107"},
        "GMMF": {"train": 95.36, "infer": 10.01, "param": "141,813,637"},
        "QARM": {"train": 306.58, "infer": 10.18, "param": "144,082,307"},
        "DMF": {"train": 72.96, "infer": 10.52, "param": "143,705,415"},
        "MARN": {"train": 139.87, "infer": 17.47, "param": "142,379,536"},
        "MTFN": {"train": 1033.04, "infer": 28.52, "param": "143,170,947"},
        "NAML": {"train": 547.79, "infer": 14.05, "param": "140,085,504"},
        "EM3": {"train": 479.64, "infer": 18.47, "param": "143,731,847"},
        "MMMLP": {"train": 737.26, "infer": 17.21, "param": "143,709,083"},
        "PAMD": {"train": 879.05, "infer": 16.09, "param": "141,503,619"},
        "M3SRec": {"train": 829.82, "infer": 19.64, "param": "143,598,227"},
        "MB": {"train": 898.71, "infer": 18.09, "param": "141,877,763"},
    },
    "(c) TikTok": {
        "Diff-MSIN": {"train": 4700.22, "infer": 98.34, "param": "9,047,328"},
        "PSRQ": {"train": 190.59, "infer": 30.74, "param": "6,698,119"},
        "LMF": {"train": 125.11, "infer": 13.11, "param": "3,354,633"},
        "MAKE": {"train": 135.40, "infer": 14.19, "param": "6,862,855"},
        "SimCEN": {"train": 163.91, "infer": 14.77, "param": "2,858,451"},
        "GMMF": {"train": 732.69, "infer": 8.15, "param": "4,454,790"},
        "QARM": {"train": 185.55, "infer": 10.30, "param": "7,362,563"},
        "DMF": {"train": 345.34, "infer": 14.46, "param": "5,917,895"},
        "MARN": {"train": 797.77, "infer": 29.52, "param": "4,626,515"},
        "MTFN": {"train": 513.28, "infer": 35.50, "param": "5,941,891"},
        "NAML": {"train": 698.61, "infer": 13.72, "param": "2,315,772"},
        "EM3": {"train": 608.64, "infer": 23.95, "param": "8,633,095"},
        "MMMLP": {"train": 1087.99, "infer": 17.87, "param": "7,484,033"},
        "PAMD": {"train": 2692.01, "infer": 26.80, "param": "4,282,755"},
        "M3SRec": {"train": 1768.59, "infer": 24.19, "param": "6,339,735"},
        "MB": {"train": 857.27, "infer": 13.76, "param": "4,042,883"},
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
datasets = ["(a) AntM²C", "(b) MicroLens", "(c) TikTok"]

# ---------------------------------------------------------------------------
# 画布：2 行 (train / infer) x 3 列 (datasets)，列内共享 x 轴
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 3, figsize=(18, 8.2),
    sharex="col", sharey="row",
    constrained_layout=True,
)
# 右侧给竖排图例留出空间
fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.06, hspace=0.05)


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


for col, dataset in enumerate(datasets):
    sub = df[df["Dataset"] == dataset].copy()

    plot_panel(axes[0, col], sub, "Train", marker="s")
    plot_panel(axes[1, col], sub, "Infer", marker="o")

    axes[0, col].set_title(dataset, fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=6)
    axes[1, col].set_xlabel("Model size (M parameters)", fontsize=AXIS_LABEL_FONT_SIZE)

axes[0, 0].set_ylabel("Training time (ms)", fontsize=AXIS_LABEL_FONT_SIZE)
axes[1, 0].set_ylabel("Inference time (ms)", fontsize=AXIS_LABEL_FONT_SIZE)

# ---------------------------------------------------------------------------
# 图例 1：形状含义（方块=训练，圆=推理）+ 类别平均星星 —— 图下方，两端对齐
# ---------------------------------------------------------------------------
shape_handles = [
    Line2D([0], [0], marker="s", color="w", label="Training time",
           markerfacecolor="gray", markeredgecolor="black", markersize=12),
    Line2D([0], [0], marker="o", color="w", label="Inference time",
           markerfacecolor="gray", markeredgecolor="black", markersize=12),
]
# 类别平均星星的图例（★ = 类别平均值）
star_handles = [
    Line2D([0], [0], marker="*", color="w", label=f"{cat} mean",
           markerfacecolor=CATEGORY_COLOR[cat], markeredgecolor="white",
           markersize=20)
    for cat in CATEGORIES
]
shape_legend = fig.legend(
    handles=shape_handles + star_handles,
    loc="lower left", bbox_to_anchor=(0.03, -0.09, 0.94, 0.05),
    ncol=6, mode="expand",
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
    loc="lower left", bbox_to_anchor=(0.03, -0.19, 0.94, 0.05),
    ncol=8, mode="expand",
    frameon=False, fontsize=LEGEND_FONT_SIZE,
    handlelength=1.4, handletextpad=0.6,
)

# ---------------------------------------------------------------------------
# 数据集之间的竖向分割线
# 先让 constrained_layout 完成最终排布，再固定下来并按真实间隙画线
# ---------------------------------------------------------------------------
fig.canvas.draw()
fig.set_layout_engine("none")  # 冻结布局，后续添加的线不会再被挪动
for col in range(len(datasets) - 1):
    bb_top_l = axes[0, col].get_position()
    bb_top_r = axes[0, col + 1].get_position()
    bb_bot = axes[1, col].get_position()
    x_sep = (bb_top_l.x1 + bb_top_r.x0) / 2.0
    line = Line2D([x_sep, x_sep], [bb_bot.y0, bb_top_l.y1],
                  transform=fig.transFigure,
                  color="#C2C2C2", linewidth=1.1, linestyle="-",
                  zorder=0.5)
    fig.add_artist(line)

save_path = OUT_DIR / "efficiency_three_datasets.pdf"
fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight",
            bbox_extra_artists=[color_legend, shape_legend])
plt.close(fig)
print(f"Saved figure: {save_path}")
