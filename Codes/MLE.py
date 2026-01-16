import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
import warnings
warnings.filterwarnings('ignore')

# 读取数据
file_path = 'Data/15/Christmas-15-Processed.xlsx'
received_df = pd.read_excel(file_path, sheet_name='Received', header=0)
total_df = pd.read_excel(file_path, sheet_name='Total', header=0)

# 数据预处理
received_df = received_df.apply(pd.to_numeric, errors='coerce')
total_df = total_df.apply(pd.to_numeric, errors='coerce')
received_df = received_df.dropna(how='all')
total_df = total_df.dropna(how='all')
received_df = received_df.iloc[:, :15]
total_df = total_df.iloc[:, :15]
received_df = received_df.reset_index(drop=True)
total_df = total_df.reset_index(drop=True)

# 确保行数一致
min_rows = min(len(received_df), len(total_df))
received_df = received_df.iloc[:min_rows]
total_df = total_df.iloc[:min_rows]

# 检查是否需要转置
if received_df.shape[0] < received_df.shape[1]:
    received_df = received_df.T
    total_df = total_df.T

# 只使用前14次抽取（第15次是确定性的）
n_draws = 14
n_red_packets = len(received_df)

print(f"数据: {n_red_packets}个红包, 每个红包{n_draws}次抽取, 共{n_red_packets * n_draws}个数据点\n")

# 构建似然函数
def negative_log_likelihood(params):
    """
    负对数似然函数
    模型: out_i ~ Beta(alpha, beta) * [c/(16-i) * in_i]
    """
    alpha, beta_param, c = params
    
    if alpha <= 0 or beta_param <= 0 or c <= 0 or c > 2:
        return np.inf
    
    log_likelihood = 0
    
    for packet_idx in range(n_red_packets):
        for i in range(n_draws):
            out_i = received_df.iloc[packet_idx, i]
            in_i = total_df.iloc[packet_idx, i]
            remaining_people = 16 - (i + 1)
            
            if remaining_people <= 0 or in_i <= 0 or out_i <= 0:
                continue
            
            max_amount = c * in_i / remaining_people
            
            if out_i >= max_amount:
                return np.inf
            
            y_i = out_i / max_amount
            
            if y_i <= 1e-10 or y_i >= 1 - 1e-10:
                return np.inf
            
            log_pdf = beta_dist.logpdf(y_i, alpha, beta_param)
            log_pdf -= np.log(max_amount)
            
            if not np.isfinite(log_pdf):
                return np.inf
            
            log_likelihood += log_pdf
    
    return -log_likelihood


# 设置初始值
initial_guesses = [
    [1.0, 1.0, 2.0],
    [2.0, 2.0, 2.0],
    [0.5, 0.5, 2.0],
    [1.0, 1.0, 1.9],
]

print("=== 开始MLE优化 ===\n")

# 优化
bounds = [(0.01, 50), (0.01, 50), (0.01, 2.0)]
results = []

for init_idx, initial_guess in enumerate(initial_guesses):
    print(f"初始值{init_idx+1}: α={initial_guess[0]}, β={initial_guess[1]}, c={initial_guess[2]}")
    try:
        result = minimize(negative_log_likelihood, initial_guess, 
                        method='L-BFGS-B', bounds=bounds, 
                        options={'maxiter': 5000, 'ftol': 1e-8, 'disp': False})
        
        if result.success:
            results.append(result)
            print(f"优化结果: α={result.x[0]:.6f}, β={result.x[1]:.6f}, c={result.x[2]:.6f}, NLL={result.fun:.4f}")
        else:
            print(f"优化结果: False")
    except:
        print(f"优化结果: False")
    print()

if len(results) == 0:
    raise ValueError("所有优化都失败了")

# 选择最佳结果
best_opt = min(results, key=lambda x: x.fun)

print("\n" + "="*60)
print("=== MLE 估计结果 ===")
print("="*60)
print(f"\n参数估计:")
print(f"  alpha = {best_opt.x[0]:.6f}")
print(f"  beta  = {best_opt.x[1]:.6f}")
print(f"  c     = {best_opt.x[2]:.6f}")
print(f"\n负对数似然值: {best_opt.fun:.4f}")

# Beta分布统计量
alpha_hat, beta_hat, c_hat = best_opt.x
mean_beta = alpha_hat / (alpha_hat + beta_hat)
var_beta = (alpha_hat * beta_hat) / ((alpha_hat + beta_hat)**2 * (alpha_hat + beta_hat + 1))

print(f"\nBeta({alpha_hat:.4f}, {beta_hat:.4f}) 统计量:")
print(f"  均值: {mean_beta:.4f}")
print(f"  标准差: {np.sqrt(var_beta):.4f}")

print(f"\n期望抽取比例 = {mean_beta:.4f} * c / 剩余人数")
print(f"  例: 剩余15人时, 期望抽取 {mean_beta * c_hat / 15:.4f} * in_i")
print("="*60)
