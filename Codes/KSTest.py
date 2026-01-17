import numpy as np
import pandas as pd
from scipy.stats import beta, ks_2samp
import matplotlib.pyplot as plt

# Estimated model parameters from MLE
alpha = 1.074343
beta_param = 1.073088  # Renamed to avoid conflict
c = 2.000000

# ----------------------------
# 1. LOAD AND PREPARE VALIDATION DATA
# ----------------------------
# Load validation data (replace with your file path)
file_path = "Data/15/Dataset-Validate.CSV"
val_data = pd.read_csv(file_path)  # Columns: 'out', 'in', 'rem'

# Compute U_i for each observation
val_data['U'] = (val_data['in'] * c) / val_data['rem']

# Normalize received amounts: y_i = out_i / U_i
val_data['y'] = val_data['out'] / val_data['U']

# Ensure all values are within [0, 1]
assert (val_data['y'] >= 0).all() and (val_data['y'] <= 1).all(), "Normalized values must be in [0, 1]."

y_real = val_data['y'].values
for i in range(len(y_real)):
        if y_real[i] < 0.05:
            y_real[i] = 0.05
n_real = len(y_real)

# ----------------------------
# 2. GENERATE SYNTHETIC DATA FROM MODEL
# ----------------------------
def generate_synthetic_data(n_samples=None, seed=42):
    """
    Generate synthetic data from the estimated Beta distribution.
    
    Returns normalized values y_synth ~ Beta(alpha, beta_param)
    """
    if n_samples is None:
        n_samples = n_real  # Match validation data size by default
    
    np.random.seed(seed)  # For reproducibility
    y_synth = beta.rvs(alpha, beta_param, size=n_samples)

    # j = 1
    for i in range(len(y_synth)):
        if y_synth[i] < 0.05:
            y_synth[i] = 0.05
            # print('modified', y_synth[i], j)
            # j+=1
        # if i%5 == 0:
            # print(y_synth[i])

    return y_synth

# Generate synthetic data
y_synth = generate_synthetic_data(n_samples=n_real, seed=632)

# ----------------------------
# 3. PERFORM TWO-SAMPLE KS TEST
# ----------------------------
ks_statistic, p_value = ks_2samp(y_real, y_synth)

# Output results
print("=" * 50)
print("TWO-SAMPLE KOLMOGOROV-SMIRNOV TEST RESULTS")
print("=" * 50)
print(f"Sample sizes: Real data = {n_real}, Synthetic data = {len(y_synth)}")
print(f"KS Statistic: {ks_statistic:.6f}")
print(f"P-value: {p_value:.6f}")
print()

# Interpretation
if p_value < 0.05:
    print("Result: Reject the null hypothesis at 5% significance level.")
    print("The real data and synthetic data appear to come from DIFFERENT distributions.")
else:
    print("Result: Fail to reject the null hypothesis at 5% significance level.")
    print("The real data and synthetic data are consistent with coming from the SAME distribution.")

# ----------------------------
# 4. VISUAL COMPARISON
# ----------------------------
def plot_ecdf_comparison(y_real, y_synth, bins=50):
    """Plot ECDFs of both datasets for visual comparison."""
    
    def ecdf(data):
        """Compute ECDF."""
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y
    
    x_real, y_ecdf_real = ecdf(y_real)
    x_synth, y_ecdf_synth = ecdf(y_synth)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ECDF plot
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
    
    # Theoretical Beta PDF
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
    
    # Additional statistics
    print("\n" + "=" * 50)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 50)
    print(f"{'Statistic':<20} {'Real Data':<15} {'Synthetic Data':<15}")
    print("-" * 50)
    print(f"{'Mean':<20} {np.mean(y_real):<15.4f} {np.mean(y_synth):<15.4f}")
    print(f"{'Std Dev':<20} {np.std(y_real):<15.4f} {np.std(y_synth):<15.4f}")
    print(f"{'Min':<20} {np.min(y_real):<15.4f} {np.min(y_synth):<15.4f}")
    print(f"{'25% Percentile':<20} {np.percentile(y_real, 25):<15.4f} {np.percentile(y_synth, 25):<15.4f}")
    print(f"{'Median':<20} {np.median(y_real):<15.4f} {np.median(y_synth):<15.4f}")
    print(f"{'75% Percentile':<20} {np.percentile(y_real, 75):<15.4f} {np.percentile(y_synth, 75):<15.4f}")
    print(f"{'Max':<20} {np.max(y_real):<15.4f} {np.max(y_synth):<15.4f}")

# Generate visual comparison
plot_ecdf_comparison(y_real, y_synth, bins=30)

# ----------------------------
# 5. MULTIPLE SIMULATIONS FOR ROBUSTNESS
# ----------------------------
def multiple_ks_simulations(y_real, n_simulations=100, seed_start=100):
    """
    Run KS test multiple times with different synthetic datasets
    to assess robustness.
    """
    ks_stats = []
    p_values = []
    
    for i in range(n_simulations):
        # Generate new synthetic dataset with different seed
        y_synth_i = generate_synthetic_data(n_samples=len(y_real), 
                                           seed=seed_start + i)
        # Perform KS test
        ks_stat_i, p_value_i = ks_2samp(y_real, y_synth_i)
        ks_stats.append(ks_stat_i)
        p_values.append(p_value_i)
    
    ks_stats = np.array(ks_stats)
    p_values = np.array(p_values)
    
    print("\n" + "=" * 50)
    print(f"ROBUSTNESS ANALYSIS ({n_simulations} simulations)")
    print("=" * 50)
    print(f"KS Statistic - Mean: {np.mean(ks_stats):.4f}, Std: {np.std(ks_stats):.4f}")
    print(f"KS Statistic - 95% CI: [{np.percentile(ks_stats, 2.5):.4f}, "
          f"{np.percentile(ks_stats, 97.5):.4f}]")
    print(f"P-value - Mean: {np.mean(p_values):.4f}, Std: {np.std(p_values):.4f}")
    print(f"Proportion of tests rejecting H0 (p < 0.05): "
          f"{np.mean(np.array(p_values) < 0.05):.2%}")
    
    # Plot distribution of KS statistics
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(ks_stats, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(ks_statistic, color='red', linestyle='--', 
                label=f'Original KS = {ks_statistic:.4f}')
    plt.xlabel('KS Statistic')
    plt.ylabel('Frequency')
    plt.title('Distribution of KS Statistics (Multiple Simulations)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(p_values, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(0.05, color='red', linestyle='--', label='α = 0.05')
    plt.xlabel('P-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of P-values (Multiple Simulations)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Optional: Run multiple simulations
multiple_ks_simulations(y_real, n_simulations=500, seed_start=632)