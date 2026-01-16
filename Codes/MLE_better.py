# 暂时不用这个代码，因为保底机制触发次数太少，而且逻辑难以描述

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 读取数据 (第1列: 剩余人数, 第2列: 剩余金额in_i, 第3列: 抽取金额out_i)
file_path = 'Data/15/Christmas15-Dataset.xlsx'
df = pd.read_excel(file_path, header=None)  

remaining_people = df.iloc[:, 0].values
in_i_values = df.iloc[:, 1].values
out_i_values = df.iloc[:, 2].values
n_data_points = len(remaining_people)

n_draws = 14
n_red_packets = n_data_points // n_draws
print(f"数据: {n_red_packets}个红包, 每个红包{n_draws}次抽取, 共{n_red_packets * n_draws}个数据点\n")

# 统计保底机制触发情况
min_amounts = in_i_values / (10 * remaining_people)
tolerance = 1e-6  # 容差，判断是否触发保底
is_floor_triggered = np.abs(out_i_values - min_amounts) < tolerance
n_floor_triggered = np.sum(is_floor_triggered)
print(f"保底机制触发: {n_floor_triggered}/{n_data_points} 次 ({100*n_floor_triggered/n_data_points:.2f}%)\n")


# 构建似然函数（考虑保底机制）
def negative_log_likelihood(params):
    """
    负对数似然函数，考虑保底机制
    模型: out_i ~ max(Beta(alpha, beta) * [c * in_i / remaining_people], floor_amount)
    其中 floor_amount = in_i / (10 * remaining_people)
    
    对于观察到的数据：
    1. 如果 out_i > floor_amount: 使用PDF（正常情况）
    2. 如果 out_i ≈ floor_amount: 使用CDF（触发保底，真实值 <= floor_amount）
    """
    alpha, beta_param, c = params
    
    if alpha <= 0 or beta_param <= 0 or c <= 0 or c > 2:
        return np.inf
    
    log_likelihood = 0
    tolerance = 1e-6  # 判断是否触发保底的容差
    
    for i in range(n_data_points):
        out_i = out_i_values[i]
        in_i = in_i_values[i]
        remaining = remaining_people[i]
        
        if remaining <= 0 or in_i <= 0 or out_i <= 0:
            continue
        
        # 计算保底金额
        floor_amount = in_i / (10 * remaining)
        
        # 最大可能抽取金额
        max_amount = c * in_i / remaining
        if out_i >= max_amount:
            return np.inf
        
        # 检查是否触发保底机制
        is_floor = abs(out_i - floor_amount) < tolerance
        
        if is_floor:
            # 触发保底：真实模型输出 <= floor_amount
            # 使用累积分布函数 P(X <= floor_amount)
            y_floor = floor_amount / max_amount
            
            # 确保y_floor在有效范围内
            if y_floor <= 0 or y_floor >= 1:
                return np.inf
            
            # 计算CDF的对数
            cdf_value = beta_dist.cdf(y_floor, alpha, beta_param)
            
            # CDF接近0会导致log(CDF)趋向负无穷，需要特殊处理
            if cdf_value < 1e-300:
                return np.inf
            
            log_prob = np.log(cdf_value)
            
        else:
            # 未触发保底：正常使用PDF
            y_i = out_i / max_amount
            
            # 确保y_i在有效范围内
            if y_i <= 1e-10 or y_i >= 1 - 1e-10:
                return np.inf
            
            # Beta分布的对数概率密度
            log_pdf = beta_dist.logpdf(y_i, alpha, beta_param)
            # 变量变换的Jacobian
            log_prob = log_pdf - np.log(max_amount)
        
        if not np.isfinite(log_prob):
            return np.inf
        
        log_likelihood += log_prob
    
    return -log_likelihood


# 设置初始值 - 在alpha和beta的不同区域尝试
initial_guesses = [
    [0.6, 0.6, 2.0],
    [0.8, 0.8, 2.0],
    [1.0, 1.0, 2.0],
    [1.2, 1.2, 2.0],
    [1.5, 1.5, 2.0],
    [2.0, 2.0, 2.0],
    [3.0, 3.0, 2.0],
    [1.0, 1.5, 2.0],
    [1.5, 1.0, 2.0],
    [1.0, 2.0, 2.0],
    [2.0, 1.0, 2.0],
    [1.0, 1.0, 1.8],
    [1.0, 1.0, 1.9],
    [1.5, 1.5, 1.9],
    [2.0, 2.0, 1.9],
]

print("=== 开始MLE优化 ===\n")

bounds = [(0.01, 50), (0.01, 50), (0.01, 2.0)]
results = []

for init_idx, initial_guess in enumerate(initial_guesses):
    print(f"初始值{init_idx+1}: α={initial_guess[0]}, β={initial_guess[1]}, c={initial_guess[2]}")
    try:
        result = minimize(negative_log_likelihood, initial_guess, 
                        method='L-BFGS-B', bounds=bounds, 
                        options={'maxiter': 15000, 'ftol': 1e-10, 'disp': False})
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
print("=== MLE 估计结果 (考虑保底机制) ===")
print("="*60)
print(f"\n参数估计:")
print(f"  alpha = {best_opt.x[0]:.6f}")
print(f"  beta  = {best_opt.x[1]:.6f}")
print(f"  c     = {best_opt.x[2]:.6f}")
print(f"\n负对数似然值: {best_opt.fun:.4f}")

# 计算一些诊断统计量
alpha_est, beta_est, c_est = best_opt.x
print(f"\n模型诊断:")
print(f"  Beta分布均值: {alpha_est/(alpha_est + beta_est):.4f}")
print(f"  Beta分布方差: {(alpha_est*beta_est)/((alpha_est+beta_est)**2*(alpha_est+beta_est+1)):.6f}")
print(f"  最大抽取倍数c: {c_est:.4f}")

# 分析保底触发对模型的影响
n_floor = np.sum(is_floor_triggered)
print(f"\n保底机制:")
print(f"  触发次数: {n_floor}/{n_data_points}")
print(f"  触发比例: {100*n_floor/n_data_points:.2f}%")


