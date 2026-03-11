import pandas as pd


data = pd.read_csv("train.csv")
df = pd.DataFrame(data)

print(df.columns)