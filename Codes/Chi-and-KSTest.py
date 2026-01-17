import numpy as np
import pandas as pd
from scipy.stats import beta, chi2, chisquare, ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
from collections import Counter

# Estimated model parameters from MLE
alpha = 1.074343
beta_param = 1.073088
c = 2.000000

# ----------------------------
# 1. LOAD AND PREPARE DATA
# ----------------------------
# Load validation data
file_path = "Data/15/Dataset-Validate.CSV"
val_data = pd.read_csv(file_path)  # Columns: 'out', 'in', 'rem'

# Compute U_i and normalize
val_data['U'] = (val_data['in'] * c) / val_data['rem']
val_data['y'] = val_data['out'] / val_data['U']

# Ensure all values are within [0, 1]
assert (val_data['y'] >= 0).all() and (val_data['y'] <= 1).all(), "Normalized values must be in [0, 1]."

y_real = val_data['y'].values
for i in range(len(y_real)):
            if y_real[i] < 0.05:
                y_real[i] = 0.05
n_real = len(y_real)

# ----------------------------
# 2. GENERATE SYNTHETIC DATA
# ----------------------------
def generate_synthetic_data(n_samples, seed=42):
    """Generate synthetic data from the model."""
    np.random.seed(seed)
    y_synth = beta.rvs(alpha, beta_param, size=n_samples)
    for i in range(len(y_synth)):
        if y_synth[i] < 0.05:
            y_synth[i] = 0.05

    return y_synth

# Generate synthetic data (same size as validation data)
y_synth = generate_synthetic_data(n_real, seed=632)

print("=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print(f"Validation data size: {n_real}")
print(f"Synthetic data size: {len(y_synth)}")
print(f"Validation data mean: {np.mean(y_real):.4f}, std: {np.std(y_real):.4f}")
print(f"Synthetic data mean: {np.mean(y_synth):.4f}, std: {np.std(y_synth):.4f}")
print()

# ----------------------------
# 3. BINNING UTILITIES
# ----------------------------
def determine_optimal_bins(data, min_expected=5, max_bins=20):
    """
    Determine optimal bins ensuring expected frequency >= min_expected.
    
    Returns:
    - bin_edges: optimal bin edges
    - n_bins: number of bins
    """
    # Start with Sturges' formula as initial estimate
    n = len(data)
    k_sturges = int(np.ceil(np.log2(n)) + 1)
    n_bins = min(max_bins, max(3, k_sturges))
    
    # Try to find bins with sufficient expected frequency
    for k in range(n_bins, 2, -1):
        # Equal-width bins
        bin_edges = np.linspace(0, 1, k + 1)
        
        # Calculate expected frequency under Beta distribution
        cdf_values = beta.cdf(bin_edges, alpha, beta_param)
        expected_probs = np.diff(cdf_values)
        expected = expected_probs * n
        
        if np.all(expected >= min_expected):
            return bin_edges, k
    
    # If no binning satisfies, use minimum bins
    return np.linspace(0, 1, 4), 3  # 3 bins minimum

def create_contingency_table(y1, y2, bin_edges):
    """
    Create 2xK contingency table for two samples.
    
    Returns:
    - contingency_table: 2D array with shape (2, n_bins)
    """
    # Bin both datasets
    hist1, _ = np.histogram(y1, bins=bin_edges)
    hist2, _ = np.histogram(y2, bins=bin_edges)
    
    # Combine into contingency table
    contingency_table = np.vstack([hist1, hist2])
    
    return contingency_table

# ----------------------------
# 4. CHI-SQUARE GOODNESS-OF-FIT TEST (One-sample)
# ----------------------------
def chi_square_gof_test(y_data, bin_edges):
    """
    Perform chi-square goodness-of-fit test for one sample against Beta distribution.
    """
    # Bin the data
    observed, _ = np.histogram(y_data, bins=bin_edges)
    n_bins = len(bin_edges) - 1
    
    # Calculate expected frequencies from Beta distribution
    cdf_values = beta.cdf(bin_edges, alpha, beta_param)
    expected_probs = np.diff(cdf_values)
    expected = expected_probs * len(y_data)
    
    # Ensure no zero expected frequencies
    expected = np.maximum(expected, 1e-10)
    
    # Calculate chi-square statistic
    chi2_stat = np.sum((observed - expected)**2 / expected)
    
    # Degrees of freedom: n_bins - 1 - number of estimated parameters
    # Since we're using external parameters, don't subtract them
    df = n_bins - 1  # Conservative approach
    
    # Calculate p-value
    p_value = 1 - chi2.cdf(chi2_stat, df)
    
    return chi2_stat, p_value, df, observed, expected

# ----------------------------
# 5. CHI-SQUARE TEST OF HOMOGENEITY (Two-sample)
# ----------------------------
def chi_square_homogeneity_test(y1, y2, bin_edges):
    """
    Perform chi-square test of homogeneity for two samples.
    Tests if two samples come from the same distribution.
    """
    # Create contingency table
    contingency_table = create_contingency_table(y1, y2, bin_edges)
    
    # Perform chi-square test of homogeneity
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    return chi2_stat, p_value, dof, contingency_table, expected

# ----------------------------
# 6. PERFORM ALL TESTS
# ----------------------------
def perform_all_chi_tests(y_real, y_synth):
    """Perform all chi-square tests and display results."""
    
    print("=" * 70)
    print("CHI-SQUARE TESTS RESULTS")
    print("=" * 70)
    
    # Determine optimal bins
    bin_edges, n_bins = determine_optimal_bins(y_real, min_expected=5)
    print(f"\nUsing {n_bins} bins with edges: {bin_edges.round(3)}")
    
    # Test 1: Goodness-of-fit for validation data
    print("\n1. GOODNESS-OF-FIT TEST (Validation Data vs Theoretical Beta)")
    print("-" * 60)
    chi2_gof_stat, chi2_gof_p, df_gof, obs_gof, exp_gof = chi_square_gof_test(y_real, bin_edges)
    print(f"   Chi-square statistic: {chi2_gof_stat:.4f}")
    print(f"   Degrees of freedom: {df_gof}")
    print(f"   P-value: {chi2_gof_p:.6f}")
    
    if chi2_gof_p < 0.05:
        print("   Result: REJECT null hypothesis - Data doesn't follow Beta distribution")
    else:
        print("   Result: FAIL TO REJECT - Data consistent with Beta distribution")
    
    # Test 2: Goodness-of-fit for synthetic data
    print("\n2. GOODNESS-OF-FIT TEST (Synthetic Data vs Theoretical Beta)")
    print("-" * 60)
    chi2_gof_synth_stat, chi2_gof_synth_p, df_gof_synth, obs_gof_synth, exp_gof_synth = chi_square_gof_test(y_synth, bin_edges)
    print(f"   Chi-square statistic: {chi2_gof_synth_stat:.4f}")
    print(f"   Degrees of freedom: {df_gof_synth}")
    print(f"   P-value: {chi2_gof_synth_p:.6f}")
    
    if chi2_gof_synth_p < 0.05:
        print("   Result: REJECT - Synthetic data doesn't follow Beta (unexpected!)")
    else:
        print("   Result: FAIL TO REJECT - Synthetic data matches Beta (as expected)")
    
    # Test 3: Test of homogeneity (two-sample)
    print("\n3. TEST OF HOMOGENEITY (Validation vs Synthetic Data)")
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
    
    # Test 4: KS test for comparison
    print("\n4. KOLMOGOROV-SMIRNOV TEST (Two-sample)")
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

# Run all tests
results = perform_all_chi_tests(y_real, y_synth)

# ----------------------------
# 7. VISUALIZE RESULTS
# ----------------------------
def visualize_chi_test_results(y_real, y_synth, results):
    """Create comprehensive visualizations of chi-square test results."""
    
    bin_edges = results['bin_edges']
    n_bins = len(bin_edges) - 1
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Histogram comparison
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
    
    # Plot 2: Goodness-of-fit observed vs expected (Validation)
    ax2 = plt.subplot(2, 3, 2)
    obs_gof, exp_gof = results['gof_real'][3], results['gof_real'][4]
    x_pos = np.arange(n_bins)
    width = 0.35
    
    ax2.bar(x_pos - width/2, obs_gof, width, label='Observed', alpha=0.7, color='blue')
    ax2.bar(x_pos + width/2, exp_gof, width, label='Expected', alpha=0.7, color='red')
    ax2.set_xlabel('Bin Index')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Goodness-of-Fit: Validation\nχ²={results["gof_real"][0]:.2f}, p={results["gof_real"][1]:.4f}')
    ax2.set_xticks(x_pos)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Goodness-of-fit observed vs expected (Synthetic)
    ax3 = plt.subplot(2, 3, 3)
    obs_gof_synth, exp_gof_synth = results['gof_synth'][3], results['gof_synth'][4]
    
    ax3.bar(x_pos - width/2, obs_gof_synth, width, label='Observed', alpha=0.7, color='orange')
    ax3.bar(x_pos + width/2, exp_gof_synth, width, label='Expected', alpha=0.7, color='red')
    ax3.set_xlabel('Bin Index')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Goodness-of-Fit: Synthetic\nχ²={results["gof_synth"][0]:.2f}, p={results["gof_synth"][1]:.4f}')
    ax3.set_xticks(x_pos)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Contingency table visualization
    ax4 = plt.subplot(2, 3, 4)
    cont_table = results['homogeneity'][3]
    expected_table = results['homogeneity'][4]
    
    # Plot observed frequencies
    x = np.arange(n_bins)
    ax4.bar(x - width/2, cont_table[0], width, label='Validation', alpha=0.7, color='blue')
    ax4.bar(x + width/2, cont_table[1], width, label='Synthetic', alpha=0.7, color='orange')
    ax4.set_xlabel('Bin Index')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Homogeneity Test\nχ²={results["homogeneity"][0]:.2f}, p={results["homogeneity"][1]:.4f}')
    ax4.set_xticks(x)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Standardized residuals for homogeneity test
    ax5 = plt.subplot(2, 3, 5)
    standardized_residuals = (cont_table - expected_table) / np.sqrt(expected_table)
    
    # Plot residuals for validation data
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
    
    # Plot 6: ECDF comparison
    ax6 = plt.subplot(2, 3, 6)
    
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y
    
    x_real, y_real_ecdf = ecdf(y_real)
    x_synth, y_synth_ecdf = ecdf(y_synth)
    
    ax6.plot(x_real, y_real_ecdf, 'b-', label='Validation', linewidth=2)
    ax6.plot(x_synth, y_synth_ecdf, 'orange', label='Synthetic', linewidth=2, linestyle='--')
    
    # Add KS statistic annotation
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
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    tests = [
        ("Goodness-of-fit (Validation)", results['gof_real'][1]),
        ("Goodness-of-fit (Synthetic)", results['gof_synth'][1]),
        ("Homogeneity Test", results['homogeneity'][1]),
        ("KS Test", results['ks'][1])
    ]
    
    for test_name, p_value in tests:
        status = "✓ PASS" if p_value >= 0.05 else "✗ FAIL"
        color = '\033[92m' if p_value >= 0.05 else '\033[91m'  # Green/Red
        reset = '\033[0m'
        print(f"{test_name:<35} p={p_value:.6f} {color}{status}{reset}")
    
    # Final interpretation
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

# Generate visualizations
visualize_chi_test_results(y_real, y_synth, results)

# ----------------------------
# 8. ROBUSTNESS CHECK
# ----------------------------
def robustness_check(y_real, n_simulations=100):
    """Check robustness of results with multiple synthetic datasets."""
    
    print("\n" + "=" * 70)
    print(f"ROBUSTNESS CHECK ({n_simulations} simulations)")
    print("=" * 70)
    
    gof_p_values = []
    homogeneity_p_values = []
    ks_p_values = []
    
    # Determine bins once
    bin_edges, _ = determine_optimal_bins(y_real, min_expected=5)
    
    for i in range(n_simulations):
        # Generate new synthetic dataset
        y_synth_i = generate_synthetic_data(len(y_real), seed=632 + i)
        
        # Perform tests
        chi2_homog_stat, chi2_homog_p, _, _, _ = chi_square_homogeneity_test(y_real, y_synth_i, bin_edges)
        ks_stat, ks_p = ks_2samp(y_real, y_synth_i)
        
        homogeneity_p_values.append(chi2_homog_p)
        ks_p_values.append(ks_p)
    
    # Analyze results
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
    axes[0].axvline(0.05, color='red', linestyle='--', label='α=0.05')
    axes[0].set_xlabel('P-value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Homogeneity Test P-values')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(ks_p_values, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(0.05, color='red', linestyle='--', label='α=0.05')
    axes[1].set_xlabel('P-value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('KS Test P-values')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run robustness check (optional)
robustness_check(y_real, n_simulations=100)