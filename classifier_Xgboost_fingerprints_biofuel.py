import pandas as pd

# 读取 Excel 文件
excel_file = 'new dataset.xlsx'  # 替换为你的 Excel 文件名
df = pd.read_excel(excel_file)

# 将 DataFrame 保存为 CSV 文件
csv_file = 'dataset.csv'
df.to_csv(csv_file, index=False)

print(f'Excel 文件已成功转换为 {csv_file}')







