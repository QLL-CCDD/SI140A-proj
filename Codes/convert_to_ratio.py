import pandas as pd
import numpy as np
import os

def convert_to_ratio_file_optimized(input_path, n_people):
    """
    优化版：计算比例(金额/剩余总额)
    改进点：强制使用 sum(axis=1) 计算当轮总额，防止读取到错误的Total列导致数据不闭环。
    """
    print(f"正在处理文件: {input_path} (人数: {n_people})")
    
    try:
        # 1. 读取数据
        df = pd.read_excel(input_path, sheet_name="Sheet1", header=None)
        
        # 截取中间的数据行（去除第一行Header和最后一行Total/End）
        # 假设文件结构是：第一行是文本，最后一行是文本或汇总
        raw_data = df.iloc[1:-1, :n_people].values.astype(float)
        
        rows, cols = raw_data.shape
        print(f" -> 读取有效数据行数: {rows}, 列数: {cols}")

        # 2. 强制使用“行求和”作为当轮总金额
        # 理由：对于Ratio分析，必须保证 sum(每个人) == Total，否则最后一个人不会是1.0
        # 这样可以忽略 Excel 文件中自带的 Total 列可能存在的误差或错位
        total_per_round = np.nansum(raw_data, axis=1)
        
        # 3. 计算比例矩阵
        ratio_matrix = np.zeros_like(raw_data)
        
        for r in range(rows):
            current_total = total_per_round[r] # 当前轮次的总金额
            remaining = current_total          # 剩余金额初始化
            
            for c in range(n_people):
                amt = raw_data[r, c]
                
                # 异常处理：金额为空 或 剩余金额也就是0了
                if np.isnan(amt) or remaining <= 1e-9: # 使用极小值处理浮点误差
                    ratio_matrix[r, c] = np.nan
                else:
                    # 核心公式：抢到的 / 此时剩余的
                    ratio = amt / remaining
                    # 修剪浮点数误差，最大限制在 1.0
                    ratio_matrix[r, c] = min(ratio, 1.0)
                
                # 扣除金额
                remaining -= amt

        # 4. 保存结果
        dir_name, file_name = os.path.split(input_path)
        base_name, ext = os.path.splitext(file_name)
        output_filename = f"{base_name}_ratio.xlsx"
        output_path = os.path.join(dir_name, output_filename)
        
        # 拼接 Start / End 保持格式一致
        output_df = pd.DataFrame(ratio_matrix)
        start_row = pd.DataFrame([["Start"] * n_people])
        end_row = pd.DataFrame([["End"] * n_people])
        
        final_df = pd.concat([start_row, output_df, end_row], ignore_index=True)
        
        final_df.to_excel(output_path, sheet_name="Sheet1", header=False, index=False)
        print(f"✅ 成功！已保存至: {output_path}")
        print(f"   (注: 最后一个人 Person {n_people} 的比例全是 1.0 是正常的数学现象)\n")
        return output_path

    except Exception as e:
        print(f"❌ 处理出错: {e}")
        return None

# ================= 运行 =================
# convert_to_ratio_file_optimized("Data/3/3人总.xls", 3)
convert_to_ratio_file_optimized("Data/15/15人 0.6.xls", 15)