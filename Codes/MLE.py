import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
import warnings
warnings.filterwarnings('ignore')

# 读取数据
file_path = 'Data/15/Christmas-15-Processed.xlsx'
received_df = pd.read_excel(file_path, sheet_name='Received', header=1)
total_df = pd.read_excel(file_path, sheet_name='Total', header=1)

# 删除所有非数字行
received_df = received_df.apply(pd.to_numeric, errors='coerce')
total_df = total_df.apply(pd.to_numeric, errors='coerce')

# 删除全为 NaN 的行
received_df = received_df.dropna(how='all')
total_df = total_df.dropna(how='all')

# 只保留前15列（对应15个人）
received_df = received_df.iloc[:, :15]
total_df = total_df.iloc[:, :15]

print("=== 数据概览 ===")
print(f"\nReceived 表形状: {received_df.shape}")
print("Received 表前几行:")
print(received_df.head())
print(f"\nTotal 表形状: {total_df.shape}")
print("Total 表前几行:")
print(total_df.head())

# 提取数据
# 每一行代表一个红包的15次抽取
# Received: out_i (第i个人领取的金额)
# Total: in_i (第i次抽取前的剩余金额)

# 只使用前14次抽取（i=1到14），因为第15次是确定性的
n_draws = 14  # 前14次抽取
n_red_packets = len(received_df)

print(f"\n红包数量: {n_red_packets}")
print(f"使用前 {n_draws} 次抽取进行估计")

# 构建似然函数
def negative_log_likelihood(params):
    """
    负对数似然函数
    
    模型: out_i ~ Beta(alpha, beta) * c/(16-i)
    其中 out_i 是第i次抽取的金额
    in_i 是第i次抽取前的剩余金额
    16-i 是剩余人数（总共15人，从1开始）
    
    Beta分布的支撑集是[0, 1]，所以实际抽取金额 = Beta(alpha, beta) * c/(16-i)
    因此，标准化后的变量 y_i = out_i / (c/(16-i)) = out_i * (16-i) / c 应该服从 Beta(alpha, beta)
    """
    alpha, beta_param, c = params
    
    # 参数约束检查
    if alpha <= 0 or beta_param <= 0 or c <= 0:
        return np.inf
    
    log_likelihood = 0
    
    for packet_idx in range(n_red_packets):
        for i in range(n_draws):
            # 第 i+1 个人抽取（索引从0开始，对应第1个人）
            out_i = received_df.iloc[packet_idx, i]
            remaining_people = 16 - (i + 1)  # 剩余人数，16-1=15, 16-2=14, ...
            
            if remaining_people <= 0:
                continue
                
            # 最大可能抽取金额
            max_amount = c / remaining_people
            
            # 标准化到 [0, 1] 区间
            y_i = out_i / max_amount
            
            # 检查是否在有效范围内
            if y_i <= 0 or y_i >= 1:
                # 处理边界情况，使用小的epsilon避免数值问题
                epsilon = 1e-10
                y_i = np.clip(y_i, epsilon, 1 - epsilon)
            
            # Beta分布的概率密度
            # PDF: f(y; alpha, beta) = y^(alpha-1) * (1-y)^(beta-1) / B(alpha, beta)
            # 对数PDF
            log_pdf = beta_dist.logpdf(y_i, alpha, beta_param)
            
            # 变量变换的雅可比行列式: dy/d(out) = 1/max_amount
            # 所以需要减去 log(max_amount)
            log_pdf -= np.log(max_amount)
            
            if np.isfinite(log_pdf):
                log_likelihood += log_pdf
            else:
                return np.inf
    
    return -log_likelihood

# 初始参数猜测
# alpha, beta 可能在1附近（均匀分布时为1）
# c 可能接近平均最大单次抽取金额
initial_guess = [1.0, 1.0, 200.0]

print("\n=== 开始优化 ===")
print(f"初始参数猜测: alpha={initial_guess[0]}, beta={initial_guess[1]}, c={initial_guess[2]}")

# 使用不同的优化方法
methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B']
results = []

for method in methods:
    print(f"\n尝试优化方法: {method}")
    try:
        if method == 'L-BFGS-B':
            # 对于 L-BFGS-B 需要设置边界
            bounds = [(0.01, 50), (0.01, 50), (1, 1000)]
            result = minimize(negative_log_likelihood, initial_guess, 
                            method=method, bounds=bounds, 
                            options={'maxiter': 10000})
        else:
            result = minimize(negative_log_likelihood, initial_guess, 
                            method=method, 
                            options={'maxiter': 10000})
        
        results.append((method, result))
        print(f"  成功: {result.success}")
        print(f"  参数估计: alpha={result.x[0]:.4f}, beta={result.x[1]:.4f}, c={result.x[2]:.4f}")
        print(f"  负对数似然: {result.fun:.2f}")
    except Exception as e:
        print(f"  失败: {e}")

# 选择最佳结果
best_result = min(results, key=lambda x: x[1].fun)
best_method, best_opt = best_result

print("\n" + "="*50)
print("=== MLE 估计结果 ===")
print("="*50)
print(f"最佳优化方法: {best_method}")
print(f"\n参数估计:")
print(f"  alpha = {best_opt.x[0]:.6f}")
print(f"  beta  = {best_opt.x[1]:.6f}")
print(f"  c     = {best_opt.x[2]:.6f}")
print(f"\n优化信息:")
print(f"  负对数似然值: {best_opt.fun:.2f}")
print(f"  优化成功: {best_opt.success}")
print(f"  迭代次数: {best_opt.nit if hasattr(best_opt, 'nit') else 'N/A'}")
print(f"  退出信息: {best_opt.message}")

# 模型诊断
print("\n=== 模型诊断 ===")
alpha_hat, beta_hat, c_hat = best_opt.x

# Beta分布的均值和方差
mean_beta = alpha_hat / (alpha_hat + beta_hat)
var_beta = (alpha_hat * beta_hat) / ((alpha_hat + beta_hat)**2 * (alpha_hat + beta_hat + 1))

print(f"Beta分布(alpha={alpha_hat:.4f}, beta={beta_hat:.4f})的统计量:")
print(f"  均值: {mean_beta:.4f}")
print(f"  方差: {var_beta:.6f}")
print(f"  标准差: {np.sqrt(var_beta):.4f}")

# 对于不同剩余人数，计算期望抽取金额
print(f"\n在 c={c_hat:.2f} 下，不同剩余人数时的期望抽取金额:")
for remaining in [15, 10, 5, 2, 1]:
    max_amount = c_hat / remaining
    expected_draw = mean_beta * max_amount
    print(f"  剩余 {remaining:2d} 人: 最大可抽 {max_amount:7.2f}, 期望抽取 {expected_draw:7.2f}")

# 计算实际数据的统计量进行对比
print("\n=== 实际数据统计 ===")
for i in range(min(5, n_draws)):
    remaining_people = 16 - (i + 1)
    out_values = received_df.iloc[:, i].values
    print(f"第 {i+1:2d} 次抽取 (剩余{remaining_people:2d}人): 均值={np.mean(out_values):7.2f}, 标准差={np.std(out_values):7.2f}")

