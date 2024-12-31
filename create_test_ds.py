import pandas as pd
import numpy as np

df = pd.read_csv("./datasets/taipei_gallery/test/taipei_500.csv")
print(df.head())

unique_sample = df.groupby(['LAT', 'LON']).apply(lambda group: group.sample(n=1, random_state=42)).reset_index(drop=True)
new_df = unique_sample.sample(n=1000)
print(new_df.head())

new_df.to_csv("./datasets/taipei_gallery/test/taipei_1000.csv", index=False)
