import pandas as pd

df = pd.read_excel("BlaBla.xlsx")
print(df.shape)
print(df.dtypes)
print(df.head())
print(df.nunique())
