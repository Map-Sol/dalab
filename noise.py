import pandas as pd
import numpy as np
from scipy import stats
data = {
    'Marks': [40, 45, 50, 55, 1000],
    'Age': [20, 21, 19, 22, 200]
}
df = pd.DataFrame(data)

print("Original Data:\n", df)

z_scores = np.abs(stats.zscore(df))

threshold = 3

mask = (z_scores < threshold).all(axis=1)

df_clean = df[mask]

print("\nAfter Removing Noise (Z-score):\n", df_clean)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_iqr = df[~((df < lower) | (df > upper)).any(axis=1)]

print("\nAfter Removing Noise (IQR):\n", df_iqr)
