import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)

x = df['sepal length (cm)']
y = df['petal length (cm)']

plt.plot(x, y)
plt.title("Line Plot")
plt.show()

plt.bar(x, y)
plt.title("Bar Plot")
plt.show()

plt.scatter(x, y)
plt.title("Scatter Plot")
plt.show()

plt.bar(df.columns, df.mean())
plt.title("Column Visualization")
plt.xticks(rotation=45)
plt.show()
