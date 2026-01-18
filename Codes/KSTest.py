import numpy as np
import pandas as pd
from scipy.stats import beta, ks_2samp
import matplotlib.pyplot as plt

# MLE Parameters
alpha = 1.000000
beta_param = 1.000000  # Renamed to avoid conflict
c = 2.000000

# ---Expected data preprocess---
file_path = "Data/15/Dataset-Validate.CSV"
val_data = pd.read_csv(file_path)  # Columns: 'rem', 'in', 'out'
val_data['U'] = (val_data['in'] * c) / val_data['rem']
val_data['y'] = val_data['out'] / val_data['U']

# Due to WeChat's round down on the money, the lower bound after normalize leaks below 0.05.
# Fix that by adding a hard filter
y_real = val_data['y'].values
for i in range(len(y_real)):
        if y_real[i] < 0.05:
            y_real[i] = 0.05
n_real = len(y_real)

# ---Model Data Gen---
def generate_synthetic_data(n_samples=None, seed=42):
    if n_samples is None:
        n_samples = n_real    
    np.random.seed(seed)
    y_synth = beta.rvs(alpha, beta_param, size=n_samples)

    # By the observed lower bound pattern, add a filter accordingly.
    for i in range(len(y_synth)):
        if y_synth[i] < 0.05:
            y_synth[i] = 0.05

    return y_synth

y_synth = generate_synthetic_data(n_samples=n_real, seed=632)

# ---KS Test two sample---
ks_statistic, p_value = ks_2samp(y_real, y_synth)

print("=" * 50)
print("TWO-SAMPLE KOLMOGOROV-SMIRNOV TEST RESULTS")
print("=" * 50)
print(f"Sample sizes: Real data = {n_real}, Synthetic data = {len(y_synth)}")
print(f"KS Statistic: {ks_statistic:.6f}")
print(f"P-value: {p_value:.6f}")
print()

if p_value < 0.05:
    print("Result: Reject the null hypothesis at 5% significance level.")
    print("The validation data and synthetic data appear to come from DIFFERENT distributions.")
else:
    print("Result: Fail to reject the null hypothesis at 5% significance level.")
    print("The validation data and synthetic data are consistent with coming from the SAME distribution.")

# ---Visualize---
def plot_ecdf_comparison(y_real, y_synth, bins=50):    
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y
    
    x_real, y_ecdf_real = ecdf(y_real)
    x_synth, y_ecdf_synth = ecdf(y_synth)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ECDF
    axes[0].plot(x_real, y_ecdf_real, label='Real Data', linewidth=2, alpha=0.8)
    axes[0].plot(x_synth, y_ecdf_synth, label='Synthetic Data', linewidth=2, alpha=0.8, linestyle='--')
    axes[0].set_xlabel('Normalized Amount (y = out/U)', fontsize=12)
    axes[0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[0].set_title(f'ECDF Comparison (KS statistic = {ks_statistic:.4f})', fontsize=14)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram comparison
    axes[1].hist(y_real, bins=bins, density=True, alpha=0.6, 
                 label='Real Data', color='blue', edgecolor='black')
    axes[1].hist(y_synth, bins=bins, density=True, alpha=0.6, 
                 label='Synthetic Data', color='orange', edgecolor='black', linestyle='--')
    
    # Theoretical Beta PDF (Uniform)
    x_pdf = np.linspace(0, 1, 1000)
    y_pdf = beta.pdf(x_pdf, alpha, beta_param)
    axes[1].plot(x_pdf, y_pdf, 'r-', linewidth=2, label=f'Beta({alpha:.2f}, {beta_param:.2f}) PDF')
    
    axes[1].set_xlabel('Normalized Amount (y = out/U)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Histogram Comparison with Theoretical PDF', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 50)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 50)
    print(f"{'Statistic':<20} {'Val Data':<15} {'Synthetic Data':<15}")
    print("-" * 50)
    print(f"{'Mean':<20} {np.mean(y_real):<15.4f} {np.mean(y_synth):<15.4f}")
    print(f"{'Std Dev':<20} {np.std(y_real):<15.4f} {np.std(y_synth):<15.4f}")
    print(f"{'Min':<20} {np.min(y_real):<15.4f} {np.min(y_synth):<15.4f}")
    print(f"{'Median':<20} {np.median(y_real):<15.4f} {np.median(y_synth):<15.4f}")
    print(f"{'Max':<20} {np.max(y_real):<15.4f} {np.max(y_synth):<15.4f}")

plot_ecdf_comparison(y_real, y_synth, bins=30)

# ---Multiple rounds for reliability---
def multiple_ks_simulations(y_real, n_simulations=100, seed_start=100):
    ks_stats = []
    p_values = []
    
    for i in range(n_simulations):
        y_synth_i = generate_synthetic_data(n_samples=len(y_real), 
                                           seed=seed_start + i)
        ks_stat_i, p_value_i = ks_2samp(y_real, y_synth_i)
        ks_stats.append(ks_stat_i)
        p_values.append(p_value_i)
    
    ks_stats = np.array(ks_stats)
    p_values = np.array(p_values)
    
    print("\n" + "=" * 50)
    print(f"ROBUSTNESS ANALYSIS ({n_simulations} simulations)")
    print("=" * 50)
    print(f"KS Statistic - Mean: {np.mean(ks_stats):.4f}, Std: {np.std(ks_stats):.4f}")
    print(f"P-value - Mean: {np.mean(p_values):.4f}, Std: {np.std(p_values):.4f}")
    print(f"Proportion of tests rejecting H0 (p < 0.05): "
          f"{np.mean(np.array(p_values) < 0.05):.2%}")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(ks_stats, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(ks_statistic, color='red', linestyle='--', 
                label=f'Original KS = {ks_statistic:.4f}')
    plt.xlabel('KS Statistic')
    plt.ylabel('Frequency')
    plt.title('Distribution of KS Statistics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(p_values, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(0.05, color='red', linestyle='--', label='alpha = 0.05')
    plt.xlabel('P-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of P-values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

multiple_ks_simulations(y_real, n_simulations=500, seed_start=632)