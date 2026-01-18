import pandas as pd
import numpy as np
import os

def convert_to_ratio_file(input_path, n_people):
    """
    读取金额表格，计算比例(金额/剩余总额)，并保存为新文件。
    新文件名将自动在原文件名后加上 "_ratio"。
    """
    # 1. 读取数据
    df = pd.read_excel(input_path, sheet_name="Sheet1", header=None)
    raw_data = df.iloc[1:-1, :n_people].values.astype(float)
    
    # 2. 确定每一轮的总金额
    # 如果表格列数比人数多，假设最后一列是总金额；否则按行求和
    if df.shape[1] > n_people:
        total_per_round = df.iloc[1:-1, n_people].values.astype(float)
    else:
        total_per_round = np.nansum(raw_data, axis=1)
    
    # 3. 计算比例矩阵
    ratio_matrix = np.zeros_like(raw_data)
    
    for r in range(len(raw_data)):
        remaining = total_per_round[r]
        for c in range(n_people):
            amt = raw_data[r, c]
            
            # 异常处理：如果是空值或剩余金额<=0
            if np.isnan(amt) or remaining <= 0:
                ratio_matrix[r, c] = np.nan
            else:
                # 计算比例，并限制最大值为1.0
                ratio = amt / remaining
                ratio_matrix[r, c] = min(ratio, 1.0)
            
            # 扣除当前金额，为下一个人做准备
            remaining -= amt

    # 4. 保存文件
    # 构造输出文件名：例如 "15人.xls" -> "15人_ratio.xlsx"
    # 注意：输出格式改为 .xlsx（现代Excel格式），因为pandas无法写入 .xls 格式
    dir_name, file_name = os.path.split(input_path)
    base_name, ext = os.path.splitext(file_name)
    output_filename = f"{base_name}_ratio.xlsx"  # 强制使用 .xlsx 格式
    output_path = os.path.join(dir_name, output_filename)
    
    # 构造新的DataFrame并保存 (保留原来格式，无表头)
    # 为了保持格式一致，我们可以创建一个空的DataFrame填充进去，或者简单地存为CSV/Excel
    output_df = pd.DataFrame(ratio_matrix)
    # 为了匹配原读取逻辑(跳过第一行和最后一行)，我们在上下各加一行空白或表头
    final_df = pd.concat([pd.DataFrame([["Start"]*n_people]), output_df, pd.DataFrame([["End"]*n_people])], ignore_index=True)
    
    final_df.to_excel(output_path, sheet_name="Sheet1", header=False, index=False)
    print(f"✅ 转换完成！比例文件已保存至: {output_path}")
    return output_path

# ================= 使用示例 =================
# 你只需要在这里修改路径和人数，运行一次即可
convert_to_ratio_file("Data/3/3人总.xls", 3)
convert_to_ratio_file("Data/15/15人.xls", 15)

