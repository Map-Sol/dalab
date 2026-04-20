import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = fetch_openml(name='weather', version=1, as_frame=True, parser='auto').frame
df = df.select_dtypes(include=np.number).dropna()
target = df.columns[-1]
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train,
y_train)
y_pred = model.predict(X_test)

print(f"\nMAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

results = pd.DataFrame({'Actual': y_test.values[:20].round(2), 'Predicted':
y_pred[:20].round(2)})
results['Error'] = (results['Actual'] - results['Predicted']).round(2)
print(f"\nWeather Forecast Table:\n{results.to_string(index=False)}")
