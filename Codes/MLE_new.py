import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist

# 读取数据 (第1列: 剩余人数, 第2列: 剩余金额in_i, 第3列: 抽取金额out_i)
file_path = "Data/15/Dataset-Train.xlsx"
df = pd.read_excel(file_path, header=None)

remaining_people = df.iloc[:, 0].values
in_i_values = df.iloc[:, 1].values
out_i_values = df.iloc[:, 2].values
n_data_points = len(remaining_people)

n_draws = 14
n_red_packets = n_data_points // n_draws
print(
    f"数据: {n_red_packets}个红包, 每个红包{n_draws}次抽取, 共{n_red_packets * n_draws}个数据点\n"
)


# 构建似然函数
def negative_log_likelihood(params):
    """
    负对数似然函数
    模型: out_i ~ Beta(alpha, beta) * [c * in_i / remaining_people]
    """
    alpha, beta_param, c = params

    if alpha <= 0 or beta_param <= 0 or c <= 0 or c > 2:
        return np.inf

    log_likelihood = 0

    for i in range(n_data_points):
        out_i = out_i_values[i]
        in_i = in_i_values[i]
        remaining = remaining_people[i]

        if remaining <= 0 or in_i <= 0 or out_i <= 0:
            continue

        # 最大可能抽取金额
        max_amount = c * in_i / remaining
        if out_i >= max_amount:
            return np.inf

        # 标准化到 [0, 1] 区间
        y_i = out_i / max_amount
        if y_i <= 1e-10 or y_i >= 1 - 1e-8:
            return np.inf

        # Beta分布的对数概率密度
        log_pdf = beta_dist.logpdf(y_i, alpha, beta_param)
        log_pdf -= np.log(max_amount)

        if not np.isfinite(log_pdf):
            return np.inf

        log_likelihood += log_pdf

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
    print(
        f"初始值{init_idx+1}: α={initial_guess[0]}, β={initial_guess[1]}, c={initial_guess[2]}"
    )
    try:
        result = minimize(
            negative_log_likelihood,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 15000, "ftol": 1e-10, "disp": False},
        )
        if result.success:
            results.append(result)
            print(
                f"优化结果: α={result.x[0]:.6f}, β={result.x[1]:.6f}, c={result.x[2]:.6f}, NLL={result.fun:.4f}"
            )
        else:
            print(f"优化结果: False")
    except:
        print(f"优化结果: False")
    print()

if len(results) == 0:
    raise ValueError("所有优化都失败了")

# 选择最佳结果
best_opt = min(results, key=lambda x: x.fun)

print("\n" + "=" * 60)
print("=== MLE 估计结果 ===")
print("=" * 60)
print(f"\n参数估计:")
print(f"  alpha = {best_opt.x[0]:.6f}")
print(f"  beta  = {best_opt.x[1]:.6f}")
print(f"  c     = {best_opt.x[2]:.6f}")
print(f"\n负对数似然值: {best_opt.fun:.4f}")
