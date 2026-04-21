import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = fetch_openml(name='weather', version=1, as_frame=True).frame
df = df.select_dtypes(include=np.number).dropna()
X, y = df.iloc[:, :-1], df.iloc[:, -1]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
pred = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr).predict(Xte)

print("MAE:", round(mean_absolute_error(yte, pred),4))
print("RMSE:", round(np.sqrt(mean_squared_error(yte, pred)),4))
print("R2:", round(r2_score(yte, pred),4))

res = pd.DataFrame({'Actual': yte.values[:10], 'Predicted': pred[:10]})
res['Error'] = res['Actual'] - res['Predicted']
print("\nWeather Forecast Table:\n", res.round(2))

plt.plot(yte.values[:50], label='Actual')
plt.plot(pred[:50], label='Predicted')
plt.legend()
plt.title("Weather Forecast Curve")
plt.xlabel("Samples")
plt.ylabel("Value")
plt.show()
