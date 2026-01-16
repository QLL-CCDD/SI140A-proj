import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, MaxNLocator

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

df = pd.read_excel(
    "C:/Users/pc/Desktop/新建 XLS 工作表 (2).xls", sheet_name="Sheet1", header=None
)

detail_data = df.iloc[1:-1, :15].values.astype(float).round(2)
total_per_round = df.iloc[1:-1, 15].values.astype(float).round(2)
n_rounds = len(detail_data)
grab_order_labels = [f"Order {i+1}" for i in range(15)]

ratio_matrix = np.zeros_like(detail_data)
for round_idx in range(n_rounds):
    total = total_per_round[round_idx]
    remaining = total
    for order_idx in range(15):
        amount = detail_data[round_idx, order_idx]
        if np.isnan(amount) or remaining <= 0:
            ratio = np.nan
        else:
            ratio = (amount / remaining).round(4)
        ratio_matrix[round_idx, order_idx] = ratio
        remaining -= amount

all_valid_ratio = ratio_matrix[~np.isnan(ratio_matrix)]
global_min_ratio = 0.0
global_max_ratio = min(1.05, np.ceil(all_valid_ratio.max() * 10) / 10)

bin_width_ratio = 0.02
bin_edges_ratio = np.arange(
    global_min_ratio, global_max_ratio + bin_width_ratio, bin_width_ratio
)
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle(
    "Ratio-Probability Histogram (More Bars: 2%/Bin) - 15 Grab Orders",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

colors = cm.Set3(np.linspace(0, 1, 15))

for idx, (ax, order_ratio, color) in enumerate(zip(axes.flat, ratio_matrix.T, colors)):

    valid_ratio = order_ratio[~np.isnan(order_ratio)]
    total_samples = len(valid_ratio)
    if total_samples == 0:
        ax.set_visible(False)
        continue

    freq, _ = np.histogram(valid_ratio, bins=bin_edges_ratio)
    prob = freq / total_samples
    bin_centers = bin_edges_ratio[:-1] + bin_width_ratio / 2

    ratio_mean = np.mean(valid_ratio).round(4)
    ratio_std = np.std(valid_ratio).round(4)
    ratio_max = np.max(valid_ratio).round(4)
    ratio_var = (ratio_std**2).round(6)

    ax.bar(
        bin_centers,
        prob,
        width=bin_width_ratio - 0.001,
        edgecolor="black",
        alpha=0.8,
        color=color,
    )

    ax.annotate(
        f"Max Ratio: {ratio_max:.4f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        fontweight="bold",
        color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        ha="left",
        va="top",
    )

    ax.set_xlim(global_min_ratio, global_max_ratio)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
    ax.xaxis.set_minor_locator(MultipleLocator(bin_width_ratio))
    ax.tick_params(axis="x", which="minor", color="#cccccc", labelsize=5)
    ax.tick_params(axis="x", which="major", labelsize=9, rotation=30)

    ax.set_ylim(0, 0.3)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"{y:.2f}"))

    ax.set_title(
        f"{grab_order_labels[idx]} (n={total_samples})", fontsize=12, fontweight="bold"
    )
    if idx >= 10:
        ax.set_xlabel(
            "Ratio (Amount / Remaining Amount Before Grab)", fontsize=10, labelpad=5
        )
    if idx % 5 == 0:
        ax.set_ylabel("Probability", fontsize=10, labelpad=5)

    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(
    top=0.93,
    hspace=0.5,
    wspace=0.25,
)

plt.savefig("dense_bars_ratio_probability.png", dpi=300, bbox_inches="tight")
plt.show()

print("=== Key Statistics (Ratio = Amount / Remaining Amount) - 15 Grab Orders ===")
order_stats = []
for i in range(15):
    valid_ratio = ratio_matrix[:, i][~np.isnan(ratio_matrix[:, i])]
    if len(valid_ratio) == 0:
        order_stats.append(
            {
                "Grab Order": f"Order {i+1}",
                "Sample Size": 0,
                "Mean Ratio": "N/A",
                "Std Ratio": "N/A",
                "Variance (Ratio²)": "N/A",
                "Max Ratio": "N/A",
            }
        )
        continue
    mean_val = np.mean(valid_ratio).round(4)
    std_val = np.std(valid_ratio).round(4)
    var_val = (std_val**2).round(6)
    max_val = np.max(valid_ratio).round(4)
    order_stats.append(
        {
            "Grab Order": f"Order {i+1}",
            "Sample Size": len(valid_ratio),
            "Mean Ratio": f"{mean_val:.4f}",
            "Std Ratio": f"{std_val:.4f}",
            "Variance (Ratio²)": f"{var_val:.6f}",
            "Max Ratio": f"{max_val:.4f}",
        }
    )

stats_df = pd.DataFrame(order_stats)
print(stats_df.to_string(index=False))
