import pandas as pd
import numpy as np

# Sample dataset
data = {'A':[1,2,np.nan,4,100],
        'B':[5,np.nan,7,8,200]}
df = pd.DataFrame(data)

print("Original Data:\n", df)

# 1. Handling Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)
print("\nAfter Handling Missing Values:\n", df)
print('ffill Mean :\n', df.ffill())
print('bFill Mean:\n', df.bfill())


# 2. Noise Removal (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))).any(axis=1)]
print("\nAfter Noise Removal:\n", df)

# 3. Remove Duplicates
df = df.drop_duplicates()
print("\nAfter Removing Duplicates:\n", df)


