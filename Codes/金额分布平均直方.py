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

grab_order_data = df.iloc[1:-1, :15].values.astype(float).round(2)
all_valid_data = grab_order_data[~np.isnan(grab_order_data)]
global_min = np.floor(all_valid_data.min())
global_max = np.ceil(all_valid_data.max())
grab_order_labels = [f"Order {i+1}" for i in range(15)]

bin_width = 0.50
bin_edges = np.arange(global_min, global_max + bin_width, bin_width)
bins = len(bin_edges) - 1
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle(
    "Amount-Probability Distribution (Unified Axes + Std) - 15 Grab Orders",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

colors = cm.Set3(np.linspace(0, 1, 15))

for idx, (ax, order_data, color) in enumerate(
    zip(axes.flat, grab_order_data.T, colors)
):
    valid_data = order_data[~np.isnan(order_data)].round(2)
    total_samples = len(valid_data)
    if total_samples == 0:
        ax.set_visible(False)
        continue

    freq, _ = np.histogram(valid_data, bins=bin_edges)
    prob = freq / total_samples
    bin_centers = bin_edges[:-1] + bin_width / 2

    order_mean = np.mean(valid_data).round(2)
    order_std = np.std(valid_data).round(2)
    order_max = np.max(valid_data).round(2)
    order_var = (order_std**2).round(4)

    ax.bar(
        bin_centers,
        prob,
        width=bin_width - 0.02,
        edgecolor="black",
        alpha=0.8,
        color=color,
    )

    ax.axvline(
        x=order_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {order_mean:.2f}\nStd: {order_std:.2f}",
    )

    ax.annotate(
        f"Max: {order_max:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=9,
        fontweight="bold",
        color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        ha="left",
        va="top",
    )

    ax.set_xlim(global_min, global_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
    ax.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax.tick_params(axis="x", which="minor", color="#cccccc", labelsize=6)
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=30)

    ax.set_ylim(0, 0.15)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"{y:.2f}"))

    ax.set_title(
        f"{grab_order_labels[idx]} (n={total_samples})", fontsize=11, fontweight="bold"
    )

    if idx >= 10:
        ax.set_xlabel("Red Packet Amount (RMB)", fontsize=9, labelpad=5)
    if idx % 5 == 0:
        ax.set_ylabel("Probability", fontsize=9, labelpad=5)

    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(0.98, 0.85))
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(
    top=0.93,
    hspace=0.5,
    wspace=0.3,
)

plt.savefig(
    "unified_axes_amount_probability_with_std.png", dpi=300, bbox_inches="tight"
)
plt.show()

print("=== Key Statistics (Exact to 0.01 RMB) - 15 Grab Orders ===")
order_stats = []
for i in range(15):
    valid_data = grab_order_data.T[i][~np.isnan(grab_order_data.T[i])].round(2)
    if len(valid_data) == 0:
        order_stats.append(
            {
                "Grab Order": f"Order {i+1}",
                "Sample Size": 0,
                "Mean (RMB)": "N/A",
                "Std (RMB)": "N/A",
                "Variance (RMB²)": "N/A",
                "Max (RMB)": "N/A",
            }
        )
        continue
    mean_val = np.mean(valid_data).round(2)
    std_val = np.std(valid_data).round(2)
    var_val = (std_val**2).round(4)
    max_val = np.max(valid_data).round(2)
    order_stats.append(
        {
            "Grab Order": f"Order {i+1}",
            "Sample Size": len(valid_data),
            "Mean (RMB)": f"{mean_val:.2f}",
            "Std (RMB)": f"{std_val:.2f}",
            "Variance (RMB²)": f"{var_val:.4f}",
            "Max (RMB)": f"{max_val:.2f}",
        }
    )

stats_df = pd.DataFrame(order_stats)
print(stats_df.to_string(index=False))
