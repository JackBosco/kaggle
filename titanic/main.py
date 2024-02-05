import getdata
import pandas as pd

df = getdata.compress('kaggle/input/train.csv')
df.to_csv('compressed.csv')
print(df)
print(df.columns)
df1 = getdata.compress('kaggle/input/test.csv')
df1.to_csv('test_compressed.csv')
print(df1)
print(df1.columns)