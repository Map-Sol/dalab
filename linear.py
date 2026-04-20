import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)

X = df[['sepal length (cm)']]
y = df['petal length (cm)']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot (Seaborn)
sns.scatterplot(x=X_train.squeeze(), y=y_train, label='Train')
sns.scatterplot(x=X_test.squeeze(), y=y_test, label='Test')
sns.lineplot(x=X_test.squeeze(), y=y_pred, label='Prediction')

plt.title("Linear Regression (Iris Dataset)")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()

# Metrics
print("m:", model.coef_[0])
print("c:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
