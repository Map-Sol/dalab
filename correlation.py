import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("Dataset:\n", df.head())
corr_matrix = df.corr()
print("\nCorrelation Matrix:\n", corr_matrix)
plt.figure()
sns.heatmap(corr_matrix, annot=True)
plt.title("Seaborn Heatmap - Iris Dataset")
plt.show()
