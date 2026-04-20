
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)

df.iloc[0, 0] = np.nan
df.iloc[1, 1] = np.nan
print("Original Data:\n", df.head())

df.fillna(df.mean(numeric_only=True), inplace=True)
print("\nAfter Handling Missing Values:\n", df.head())
print('ffill Mean :\n', df.ffill())
print('bFill Mean:\n', df.bfill())

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))).any(axis=1)]
print("\nAfter Noise Removal:\n", df)

df = df.drop_duplicates()
print("\nAfter Removing Duplicates:\n", df)


