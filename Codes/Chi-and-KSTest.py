import numpy as np
import pandas as pd
from scipy.stats import beta, chi2, chisquare, ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
from collections import Counter

# Estimated model parameters
alpha = 1.000000
beta_param = 1.000000
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

# ---Data Gen---
def generate_synthetic_data(n_samples, seed=42):
    np.random.seed(seed)
    y_synth = beta.rvs(alpha, beta_param, size=n_samples)
    # By the observed lower bound pattern, add a filter accordingly.
    for i in range(len(y_synth)):
        if y_synth[i] < 0.05:
            y_synth[i] = 0.05

    return y_synth

y_synth = generate_synthetic_data(n_real, seed=632)

print("=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print(f"Validation data size: {n_real}")
print(f"Synthetic data size: {len(y_synth)}")
print(f"Validation mean: {np.mean(y_real):.4f}, std: {np.std(y_real):.4f}")
print(f"Synthetic mean: {np.mean(y_synth):.4f}, std: {np.std(y_synth):.4f}")
print()

# ---Chi prepare---
def determine_optimal_bins(data, min_expected=5, max_bins=20):
    """Determine optimal bins ensuring expected frequency >= min_expected."""
    n = len(data)
    k_sturges = int(np.ceil(np.log2(n)) + 1)
    n_bins = min(max_bins, max(3, k_sturges))
    
    # Try to find bins with sufficient expected frequency
    for k in range(n_bins, 2, -1):
        bin_edges = np.linspace(0, 1, k + 1)
        
        cdf_values = beta.cdf(bin_edges, alpha, beta_param)
        expected_probs = np.diff(cdf_values)
        expected = expected_probs * n
        
        if np.all(expected >= min_expected):
            return bin_edges, k
    
    # If no binning satisfies, use minimum 3 bins
    return np.linspace(0, 1, 4), 3

def create_contingency_table(y1, y2, bin_edges):
    """Create 2*K contingency table for two samples."""
    hist1, _ = np.histogram(y1, bins=bin_edges)
    hist2, _ = np.histogram(y2, bins=bin_edges)
    
    contingency_table = np.vstack([hist1, hist2])
    
    return contingency_table

# ---Chi Squre Goodness-of-fit---
def chi_square_gof_test(y_data, bin_edges):
    """Perform chi-square gof test for one sample against Beta distribution."""
    observed, _ = np.histogram(y_data, bins=bin_edges)
    n_bins = len(bin_edges) - 1
    
    cdf_values = beta.cdf(bin_edges, alpha, beta_param)
    expected_probs = np.diff(cdf_values)
    expected = expected_probs * len(y_data)
    expected = np.maximum(expected, 1e-10)
    chi2_stat = np.sum((observed - expected)**2 / expected)
    df = n_bins - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    
    return chi2_stat, p_value, df, observed, expected

# ---Chi Square Homogeneity Test---
def chi_square_homogeneity_test(y1, y2, bin_edges):
    contingency_table = create_contingency_table(y1, y2, bin_edges)
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    return chi2_stat, p_value, dof, contingency_table, expected

# ---Chi Gof, Chi Homo, KS test perform---
def perform_all_chi_tests(y_real, y_synth):
    """Null hypothesis **H0**: "Data likely from same distribution."""
    
    print("=" * 70)
    print("CHI-SQUARE TESTS RESULTS")
    print("=" * 70)
    
    bin_edges, n_bins = determine_optimal_bins(y_real, min_expected=5)
    print(f"\nUsing {n_bins} bins with edges: {bin_edges.round(3)}")
    
    # Test 1
    print("\n1. GOF TEST (Validation vs Theoretical)")
    print("-" * 60)
    chi2_gof_stat, chi2_gof_p, df_gof, obs_gof, exp_gof = chi_square_gof_test(y_real, bin_edges)
    print(f"   Chi-square statistic: {chi2_gof_stat:.4f}")
    print(f"   Degrees of freedom: {df_gof}")
    print(f"   P-value: {chi2_gof_p:.6f}")
    
    if chi2_gof_p < 0.05:
        print("   Result: REJECT null hypothesis - Data doesn't follow Beta distribution")
    else:
        print("   Result: FAIL TO REJECT - Data consistent with Beta distribution")
    
    # Test 2
    print("\n2. GOF TEST (Synthetic vs Theoretical)")
    print("-" * 60)
    chi2_gof_synth_stat, chi2_gof_synth_p, df_gof_synth, obs_gof_synth, exp_gof_synth = chi_square_gof_test(y_synth, bin_edges)
    print(f"   Chi-square statistic: {chi2_gof_synth_stat:.4f}")
    print(f"   Degrees of freedom: {df_gof_synth}")
    print(f"   P-value: {chi2_gof_synth_p:.6f}")
    
    if chi2_gof_synth_p < 0.05:
        print("   Result: REJECT - Synthetic data doesn't follow Beta (unexpected!)")
    else:
        print("   Result: FAIL TO REJECT - Synthetic data matches Beta (as expected)")
    
    # Test 3
    print("\n3. HOMOGENEITY TEST (Validation vs Synthetic)")
    print("-" * 60)
    chi2_homog_stat, chi2_homog_p, df_homog, cont_table, expected_homog = chi_square_homogeneity_test(y_real, y_synth, bin_edges)
    print(f"   Chi-square statistic: {chi2_homog_stat:.4f}")
    print(f"   Degrees of freedom: {df_homog}")
    print(f"   P-value: {chi2_homog_p:.6f}")
    print(f"   Contingency table shape: {cont_table.shape}")
    
    if chi2_homog_p < 0.05:
        print("   Result: REJECT - Samples come from different distributions")
    else:
        print("   Result: FAIL TO REJECT - Samples likely from same distribution")
    
    # Test 4
    print("\n4. KS TEST (Two-sample)")
    print("-" * 60)
    ks_stat, ks_p = ks_2samp(y_real, y_synth)
    print(f"   KS statistic: {ks_stat:.6f}")
    print(f"   P-value: {ks_p:.6f}")
    
    if ks_p < 0.05:
        print("   Result: REJECT - Distributions are different")
    else:
        print("   Result: FAIL TO REJECT - Distributions are similar")
    
    return {
        'bin_edges': bin_edges,
        'gof_real': (chi2_gof_stat, chi2_gof_p, df_gof, obs_gof, exp_gof),
        'gof_synth': (chi2_gof_synth_stat, chi2_gof_synth_p, df_gof_synth, obs_gof_synth, exp_gof_synth),
        'homogeneity': (chi2_homog_stat, chi2_homog_p, df_homog, cont_table, expected_homog),
        'ks': (ks_stat, ks_p)
    }

results = perform_all_chi_tests(y_real, y_synth)

# ---Visualize---
def visualize_chi_test_results(y_real, y_synth, results):    
    bin_edges = results['bin_edges']
    n_bins = len(bin_edges) - 1

    fig = plt.figure(figsize=(16, 10))
    
    # Histogram comparison
    ax1 = plt.subplot(2, 3, 1)
    x = np.linspace(0, 1, 1000)
    y_pdf = beta.pdf(x, alpha, beta_param)
    
    ax1.hist(y_real, bins=bin_edges, density=True, alpha=0.5, 
             color='blue', label='Validation', edgecolor='black')
    ax1.hist(y_synth, bins=bin_edges, density=True, alpha=0.5,
             color='orange', label='Synthetic', edgecolor='black', hatch='//')
    ax1.plot(x, y_pdf, 'r-', linewidth=2, label='Beta PDF')
    ax1.set_xlabel('Normalized Amount')
    ax1.set_ylabel('Density')
    ax1.set_title('Histogram Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gof observed vs expected model
    ax2 = plt.subplot(2, 3, 2)
    obs_gof, exp_gof = results['gof_real'][3], results['gof_real'][4]
    x_pos = np.arange(n_bins)
    width = 0.35
    
    ax2.bar(x_pos - width/2, obs_gof, width, label='Observed', alpha=0.7, color='blue')
    ax2.bar(x_pos + width/2, exp_gof, width, label='Expected', alpha=0.7, color='red')
    ax2.set_xlabel('Bin Index')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Goodness-of-Fit: Validation\nX²={results["gof_real"][0]:.2f}, p={results["gof_real"][1]:.4f}')
    ax2.set_xticks(x_pos)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gof synthetic vs expected model
    ax3 = plt.subplot(2, 3, 3)
    obs_gof_synth, exp_gof_synth = results['gof_synth'][3], results['gof_synth'][4]
    
    ax3.bar(x_pos - width/2, obs_gof_synth, width, label='Observed', alpha=0.7, color='orange')
    ax3.bar(x_pos + width/2, exp_gof_synth, width, label='Expected', alpha=0.7, color='red')
    ax3.set_xlabel('Bin Index')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Goodness-of-Fit: Synthetic\nX²={results["gof_synth"][0]:.2f}, p={results["gof_synth"][1]:.4f}')
    ax3.set_xticks(x_pos)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Contingency table visualization
    ax4 = plt.subplot(2, 3, 4)
    cont_table = results['homogeneity'][3]
    expected_table = results['homogeneity'][4]
    
    x = np.arange(n_bins)
    ax4.bar(x - width/2, cont_table[0], width, label='Validation', alpha=0.7, color='blue')
    ax4.bar(x + width/2, cont_table[1], width, label='Synthetic', alpha=0.7, color='orange')
    ax4.set_xlabel('Bin Index')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Homogeneity Test\nX²={results["homogeneity"][0]:.2f}, p={results["homogeneity"][1]:.4f}')
    ax4.set_xticks(x)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Standardized residuals for homogeneity test
    ax5 = plt.subplot(2, 3, 5)
    standardized_residuals = (cont_table - expected_table) / np.sqrt(expected_table)
    
    ax5.bar(x - width/2, standardized_residuals[0], width, label='Validation', alpha=0.7, color='blue')
    ax5.bar(x + width/2, standardized_residuals[1], width, label='Synthetic', alpha=0.7, color='orange')
    ax5.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='±2σ')
    ax5.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Bin Index')
    ax5.set_ylabel('Standardized Residual')
    ax5.set_title('Standardized Residuals (Homogeneity)')
    ax5.set_xticks(x)
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # KS ECDF comparison
    ax6 = plt.subplot(2, 3, 6)
    
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y
    
    x_real, y_real_ecdf = ecdf(y_real)
    x_synth, y_synth_ecdf = ecdf(y_synth)
    
    ax6.plot(x_real, y_real_ecdf, 'b-', label='Validation', linewidth=2)
    ax6.plot(x_synth, y_synth_ecdf, 'orange', label='Synthetic', linewidth=2, linestyle='--')
    
    ks_stat, ks_p = results['ks']
    ax6.text(0.05, 0.9, f'KS={ks_stat:.4f}\np={ks_p:.4f}', 
             transform=ax6.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax6.set_xlabel('Normalized Amount')
    ax6.set_ylabel('Cumulative Probability')
    ax6.set_title('ECDF Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print primary conclusion
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    tests = [
        ("Gof Validation", results['gof_real'][1]),
        ("Gof Synthetic", results['gof_synth'][1]),
        ("Homogeneity", results['homogeneity'][1]),
        ("KS Test", results['ks'][1])
    ]
    
    for test_name, p_value in tests:
        status = "PASS" if p_value >= 0.05 else "FAIL"
        color = '\033[92m' if p_value >= 0.05 else '\033[91m'  # Green/Red
        reset = '\033[0m'
        print(f"{test_name:<35} p={p_value:.6f} {color}{status}{reset}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    gof_real_p = results['gof_real'][1]
    homogeneity_p = results['homogeneity'][1]
    
    if gof_real_p >= 0.05 and homogeneity_p >= 0.05:
        print("Both tests suggest the model fits well:")
        print("1. Validation data follows Beta distribution")
        print("2. Validation and synthetic data come from same distribution")
        print("Conclusion: Model is a good fit for the data.")
    elif gof_real_p < 0.05 and homogeneity_p >= 0.05:
        print("Conflicting results:")
        print("1. Goodness-of-fit rejects (validation doesn't match Beta)")
        print("2. Homogeneity accepts (validation and synthetic are similar)")
        print("Possible: Both datasets deviate from Beta in similar ways.")
        print("Conclusion: Model captures relative patterns but not exact distribution.")
    elif gof_real_p >= 0.05 and homogeneity_p < 0.05:
        print("Conflicting results:")
        print("1. Goodness-of-fit accepts (validation matches Beta)")
        print("2. Homogeneity rejects (validation and synthetic differ)")
        print("Possible: Synthetic data doesn't perfectly follow Beta due to sampling variability.")
        print("Conclusion: Model is theoretically correct but synthetic data has sampling error.")
    else:
        print("Both tests reject the null hypotheses:")
        print("1. Validation data doesn't follow Beta distribution")
        print("2. Validation and synthetic data come from different distributions")
        print("Conclusion: Model may not be a good fit for the data.")

visualize_chi_test_results(y_real, y_synth, results)

# ---Multiple Rounds ensure reliability---
def robustness_check(y_real, n_simulations=100):    
    print("\n" + "=" * 70)
    print(f"ROBUSTNESS CHECK ({n_simulations} simulations)")
    print("=" * 70)
    
    gof_p_values = []
    homogeneity_p_values = []
    ks_p_values = []
    
    bin_edges, _ = determine_optimal_bins(y_real, min_expected=5)
    
    for i in range(n_simulations):
        y_synth_i = generate_synthetic_data(len(y_real), seed=632 + i)

        chi2_homog_stat, chi2_homog_p, _, _, _ = chi_square_homogeneity_test(y_real, y_synth_i, bin_edges)
        homogeneity_p_values.append(chi2_homog_p)
        ks_stat, ks_p = ks_2samp(y_real, y_synth_i)
        ks_p_values.append(ks_p)
    
    print(f"\nHomogeneity test:")
    print(f"  Mean p-value: {np.mean(homogeneity_p_values):.4f}")
    print(f"  Std p-value: {np.std(homogeneity_p_values):.4f}")
    print(f"  Rejection rate (p < 0.05): {np.mean(np.array(homogeneity_p_values) < 0.05):.2%}")
    
    print(f"\nKS test:")
    print(f"  Mean p-value: {np.mean(ks_p_values):.4f}")
    print(f"  Std p-value: {np.std(ks_p_values):.4f}")
    print(f"  Rejection rate (p < 0.05): {np.mean(np.array(ks_p_values) < 0.05):.2%}")
    
    # Plot distribution of p-values
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    
    axes[0].hist(homogeneity_p_values, bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(0.05, color='red', linestyle='--', label='alpha=0.05')
    axes[0].set_xlabel('P-value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Homogeneity Test P-values')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(ks_p_values, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(0.05, color='red', linestyle='--', label='alpha=0.05')
    axes[1].set_xlabel('P-value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('KS Test P-values')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

robustness_check(y_real, n_simulations=100)