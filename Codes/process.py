import pandas as pd
import numpy as np
from openpyxl import load_workbook

# 读取Excel文件
file_path = r'C:\Users\33539\Desktop\SI140A-proj\Data\15\Christmas-15.xlsx'

# 读取Excel文件
df = pd.read_excel(file_path, sheet_name=0, header=None)

print(f"文件包含 {df.shape[0]} 行和 {df.shape[1]} 列")

# 根据输出，数据结构如下：
# 第0行：注释
# 第1行：标题行（姓名、序号\轮次、各个人的名字、总计）
# 第2行：标签行（钱数、1、...、60）或可能是数据第1行
# 需要动态判断数据起始行

# 找到数据起始行 - 从第3行（索引2）开始检查
data_start_row = None
for idx in range(2, min(5, len(df))):
    second_col = df.iloc[idx, 1]
    try:
        val = float(second_col)
        # 如果第二列是1或者是小的数字（1-10），很可能是序号的开始
        if val >= 1 and val <= 10:
            data_start_row = idx
            break
    except:
        continue

if data_start_row is None:
    data_start_row = 3

print(f"数据起始行（索引）: {data_start_row}, 即第{data_start_row+1}行")
print(f"起始行的第二列值: {df.iloc[data_start_row, 1]}")

data_end_row = None

# 找到数据结束行（通常是遇到"总计"或其他汇总行）
for idx in range(data_start_row, len(df)):
    first_col_val = str(df.iloc[idx, 0])
    # 检查第二列（序号列）是否为NaN或非数字，这表示数据行结束
    second_col_val = df.iloc[idx, 1]
    if '总计' in first_col_val or '平均' in first_col_val or pd.isna(second_col_val):
        # 如果第二列是数字，说明还是数据行
        try:
            if pd.notna(second_col_val):
                num = float(second_col_val)
                if num > 0 and num <= 200:  # 序号应该在合理范围内
                    continue
        except:
            pass
        data_end_row = idx
        break

if data_end_row is None:
    data_end_row = len(df)

print(f"数据行范围: 第{data_start_row+1}行 到 第{data_end_row}行")
print(f"数据行总数: {data_end_row - data_start_row}")

# 提取数据部分（15个人的数据在第2-16列，索引1-15）
# 第0列是标签，第1列是序号，第2-16列是15个人的数据，第17列是总计
person_start_col = 2
person_end_col = 17  # 15个人

# 先检查序号列，看看到底有多少轮数据
seq_col = df.iloc[data_start_row:data_end_row, 1].apply(pd.to_numeric, errors='coerce')
valid_seq = seq_col.dropna()
print(f"序号列的有效值范围: {valid_seq.min()} 到 {valid_seq.max()}")
print(f"序号列的有效值个数: {len(valid_seq)}")

# 提取数值数据
data_df = df.iloc[data_start_row:data_end_row, person_start_col:person_end_col].apply(pd.to_numeric, errors='coerce')

# 检查每一行是否有有效数据
valid_data_rows = data_df.notna().any(axis=1)
print(f"有有效数据的行数: {valid_data_rows.sum()}")

# 取最后150行数据（如果不足150行则全部取）
if len(data_df) > 150:
    data_df = data_df.iloc[-150:]

print(f"分析的数据行数: {len(data_df)}")
print(f"分析的人数（列数）: {len(data_df.columns)}")

# 打印几行数据来检查
print("\n前3行数据:")
print(data_df.head(3))
print("\n后3行数据:")
print(data_df.tail(3))

# 计算每个人（每列）获得最大值的次数
max_counts = []
for col_idx in range(len(data_df.columns)):
    count = 0
    for row_idx in data_df.index:
        row_values = data_df.loc[row_idx, :]
        max_value = row_values.max()
        # 检查当前列的值是否等于该行的最大值（处理可能的并列情况）
        if not pd.isna(max_value) and data_df.iloc[row_idx - data_df.index[0], col_idx] == max_value:
            count += 1
    max_counts.append(count)

print(f"\n每个人获得最大值的次数: {max_counts}")
print(f"总和验证（应该>=150，因为可能有并列）: {sum(max_counts)}")

# 检查是否已经存在"最大值出现次数"行
has_max_count_row = False
max_count_row_idx = None

for idx in range(data_end_row, len(df)):
    first_col_val = str(df.iloc[idx, 0])
    if '最大值出现次数' in first_col_val:
        has_max_count_row = True
        max_count_row_idx = idx
        print(f"\n发现已存在的'最大值出现次数'行在第{idx+1}行，将更新该行")
        break

# 更新或添加"最大值出现次数"行
if has_max_count_row:
    # 更新现有行
    df.iloc[max_count_row_idx, 0] = '最大值出现次数'
    for i, count in enumerate(max_counts):
        df.iloc[max_count_row_idx, person_start_col + i] = count
else:
    # 添加新行
    print("\n未发现'最大值出现次数'行，将添加新行")
    new_row = [None] * len(df.columns)
    new_row[0] = '最大值出现次数'
    for i, count in enumerate(max_counts):
        new_row[person_start_col + i] = count
    df.loc[len(df)] = new_row

# 使用openpyxl保存以保留格式
with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
    df.to_excel(writer, sheet_name='Sheet1', header=False, index=False)

print(f"\n文件已更新: {file_path}")
print("请检查Excel文件中的'最大值出现次数'行")

