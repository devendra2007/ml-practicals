# ml-practicals
plt.show()


#%% EXPERIMENT 20: PERCEPTRON (AND GATE)
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
weights = np.zeros(2)
bias, lr = 0.0, 0.1

for epoch in range(10):
    for i in range(len(X)):
        pred = int(np.dot(X[i], weights) + bias >= 0)
        error = y[i] - pred
        weights += lr * error * X[i]
        bias += lr * error

print("Weights:", weights)
print("Bias:", bias)
print("AND Gate Predictions:")
for x, actual in zip(X, y):
    pred = int(np.dot(x, weights) + bias >= 0)
    print(f"{x} -> Predicted: {pred}, Actual: {actual}")


#%% EXPERIMENT 21: MLP CLASSIFIER (RELU VS TANH)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for activation in ["relu", "tanh"]:
    model = MLPClassifier(hidden_layer_sizes=(16, 8), activation=activation,
                          max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print(f"MLP ({activation}) Accuracy: "
          f"{accuracy_score(y_test, model.predict(X_test)):.4f}")


#%% EXPERIMENT 22: RANDOM FOREST VS ADABOOST
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
ada = AdaBoostClassifier(random_state=42).fit(X_train, y_train)

print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
print("AdaBoost Accuracy:", accuracy_score(y_test, ada.predict(X_test)))

importance = pd.Series(rf.feature_importances_,
                       index=data.feature_names).nlargest(10)
importance.sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()


#%% EXPERIMENT 23: CLUSTERING (K-MEANS, AGGLOMERATIVE, GMM)
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

models = [
    ("K-Means", KMeans(n_clusters=4, random_state=42, n_init=10)),
    ("Agglomerative", AgglomerativeClustering(n_clusters=4)),
    ("Gaussian Mixture", GaussianMixture(n_components=4, random_state=42))
]

for ax, (name, model) in zip(axes, models):
    labels = model.fit_predict(X) if name != "Gaussian Mixture" else model.fit(X).predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
    ax.set_title(name)

plt.tight_layout()
plt.show()
