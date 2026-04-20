import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Pred:", y_pred[:10])
plt.scatter(X_test[:, 0], y_test)
plt.scatter(X_test[:, 0], y_pred, marker='x')
plt.show()
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

tree = model.estimators_[7]   # first decision tree
plot_tree(tree)
plt.show()
