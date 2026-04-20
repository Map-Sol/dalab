import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('healthexp')

print("Shape:", df.shape)
print("\nFirst Rows:\n", df.head())
print("\nStatistics:\n", df.describe())

plt.hist(df['Life_Expectancy'])
plt.title("Life Expectancy Distribution")
plt.show()

plt.plot(df['Year'], df['Spending_USD'])
plt.title("Health Spending Over Years")
plt.show()

plt.bar(df.select_dtypes(include='number').columns,
        df.select_dtypes(include='number').mean())
plt.title("Column-wise Mean Values")
plt.xticks(rotation=45)
plt.show()
