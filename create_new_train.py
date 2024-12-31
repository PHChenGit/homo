import pandas as pd

# 假設資料
data = pd.read_csv("./datasets/taipei_gallery/val/taipei.csv")
data_df = pd.DataFrame(data)
print(data_df.head())

# 過濾掉 head = 0 的資料
filtered_data = data_df[data_df['HEAD'] != 0]

# 按照 (LAT, LON) 分組，並從每組中隨機抽取一筆資料
unique_sample = filtered_data.groupby(['LAT', 'LON']).apply(lambda group: group.sample(n=1, random_state=42)).reset_index(drop=True)

# 假設大數據情況下，從已篩選的唯一組中抽取 500 筆資料
final_sample = unique_sample.sample(n=min(12000, len(unique_sample)), random_state=42)

# 保存結果
output_path = './datasets/taipei_gallery/val/taipei_12000.csv'  # 替換為你的輸出文件路徑
final_sample.to_csv(output_path, index=False)

print(f"Sampled data saved to {output_path}")
