import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)


print("MAE :", round(mean_absolute_error(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("R²  :", round(r2_score(y_test, y_pred), 4))

res = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10]
})
res['Error'] = res['Actual'] - res['Predicted']

print("\nWeather Forecast Table:\n", res.round(2))


plt.plot(y_test.values[:10], label='Actual')
plt.plot(y_pred[:10], label='Predicted')
plt.legend()
plt.title("Weather Forecast Curve")
plt.xlabel("Samples")
plt.ylabel("Value")
plt.show()
