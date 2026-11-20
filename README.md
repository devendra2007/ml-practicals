#%% EXPERIMENT 1: EXPLORATORY DATA ANALYSIS (TITANIC)
import pandas as pd
from sklearn.datasets import fetch_openml

titanic = fetch_openml("titanic", version=1, as_frame=True).frame
print("Data Types:\n", titanic.dtypes)
print("\nMissing Values:\n", titanic.isnull().sum())
print("\nDuplicates:", titanic.duplicated().sum())
print("\nStatistical Summary:\n", titanic.describe(include="all"))


#%% EXPERIMENT 2: DATA VISUALIZATION (IRIS)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

df = load_iris(as_frame=True).frame
plt.figure(figsize=(12, 9))
plt.subplot(2, 2, 1)
sns.histplot(df["sepal length (cm)"], kde=True)
plt.title("Sepal Length Histogram")

plt.subplot(2, 2, 2)
sns.boxplot(data=df.drop(columns="target"))
plt.title("Feature Box Plot")

plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="target")
plt.title("Scatter Plot")

plt.subplot(2, 2, 4)
sns.heatmap(df.drop(columns="target").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


#%% EXPERIMENT 3: FEATURE ENGINEERING (ADULT INCOME)
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

adult = fetch_openml(data_id=1590, as_frame=True).frame.sample(1000, random_state=42)
X = adult.drop(columns="class").copy()

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

X[num_cols] = SimpleImputer(strategy="mean").fit_transform(X[num_cols])
X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])

encoded = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cats = pd.DataFrame(encoded.fit_transform(X[cat_cols]),
                            columns=encoded.get_feature_names_out(cat_cols))
X_norm = MinMaxScaler().fit_transform(X[num_cols])
X_std = StandardScaler().fit_transform(X[num_cols])

print("Encoded categorical data:\n", encoded_cats.head())
print("Normalized numeric data:\n", X_norm[:2])
print("Standardized numeric data:\n", X_std[:2])


#%% EXPERIMENT 4: SIMPLE LINEAR REGRESSION (MANUAL)
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(1, 11)
y = np.array([38000, 42000, 55000, 61000, 68000,
              83000, 92000, 105000, 112000, 122000])

xm, ym = X.mean(), y.mean()
b1 = np.sum((X - xm) * (y - ym)) / np.sum((X - xm) ** 2)
b0 = ym - b1 * xm
y_pred = b0 + b1 * X

rmse = np.sqrt(np.mean((y - y_pred) ** 2))
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - ym) ** 2)
print(f"RMSE: {rmse:.2f}, R2: {r2:.4f}")

plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, "r", label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()


#%% EXPERIMENT 5: MULTIPLE LINEAR REGRESSION
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
residuals = y_test - pred

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test, pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

plt.subplot(1, 2, 2)
plt.scatter(pred, residuals, alpha=0.3)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()

#%% EXPERIMENT 6: SIMPLE BINARY CLASSIFICATION (MANUAL BOUNDARY)
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

data = load_breast_cancer()
X = data.data[:, 0]       # mean radius
y = data.target           # 0 = malignant, 1 = benign
y_pred = np.where(X < 15, 1, 0)

print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("Accuracy:", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall:", recall_score(y, y_pred))
print("F1 Score:", f1_score(y, y_pred))


#%% EXPERIMENT 7: DECISION TREE CLASSIFIER
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names,
          class_names=iris.target_names)
plt.title("Decision Tree (Iris)")
plt.show()

#%% EXPERIMENT 8: OVERFITTING IN DECISION TREES
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

depths = range(1, 20)
train_acc, test_acc = [], []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, clf.predict(X_test)))

plt.plot(depths, train_acc, label="Train Accuracy")
plt.plot(depths, test_acc, label="Test Accuracy")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting in Decision Trees")
plt.legend()
plt.show()


#%% EXPERIMENT 9: K-NEAREST NEIGHBORS (KNN)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    print(f"K={k}, Accuracy={accuracy_score(y_test, knn.predict(X_test)):.4f}")


#%% EXPERIMENT 10: KNN HYPERPARAMETER TUNING
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

grid = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": range(1, 15)}, cv=5)
grid.fit(X_train, y_train)
print("Best K:", grid.best_params_["n_neighbors"])
print("Best CV Accuracy:", grid.best_score_)
print("Test Accuracy:", grid.score(X_test, y_test))

#%% EXPERIMENT 11: CURSE OF DIMENSIONALITY
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

features = [2, 10, 50, 100, 500, 1000]
accuracies = []
for n in features:
    X, y = make_classification(n_samples=500, n_features=n,
                               n_informative=2, n_redundant=0,
                               n_repeated=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, knn.predict(X_test)))

plt.plot(features, accuracies, "o-")
plt.xscale("log")
plt.xlabel("Number of Features (log scale)")
plt.ylabel("KNN Accuracy")
plt.title("Curse of Dimensionality")
plt.grid(True)
plt.show()


#%% EXPERIMENT 12: PRINCIPAL COMPONENT ANALYSIS (PCA)
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

digits = load_digits()
X_pca = PCA(n_components=2).fit_transform(digits.data)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, cmap="tab10", alpha=0.6)
plt.colorbar(ticks=range(10))
plt.title("PCA of Digits (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


#%% EXPERIMENT 13: t-SNE DIMENSIONALITY REDUCTION
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

digits = load_digits()
X, y = digits.data[:500], digits.target[:500]
X_tsne = TSNE(n_components=2, random_state=42,
              init="pca", learning_rate="auto").fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", alpha=0.6)
plt.colorbar(ticks=range(10))
plt.title("t-SNE of Digits (2D)")
plt.show()


#%% EXPERIMENT 14: CLASSIFICATION VS PCA FEATURE REDUCTION
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = RandomForestClassifier(random_state=42).fit(X_train_s, y_train)
acc_orig = accuracy_score(y_test, clf.predict(X_test_s))

pca = PCA(n_components=5).fit(X_train_s)
X_train_pca = pca.transform(X_train_s)
X_test_pca = pca.transform(X_test_s)
clf_pca = RandomForestClassifier(random_state=42).fit(X_train_pca, y_train)
acc_pca = accuracy_score(y_test, clf_pca.predict(X_test_pca))

print(f"Accuracy with Original Features: {acc_orig:.4f}")
print(f"Accuracy with PCA (5 components): {acc_pca:.4f}")


#%% EXPERIMENT 15: USER-BASED COLLABORATIVE FILTERING
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.DataFrame({
    "Toy Story": [5, 4, 1, 0, 0],
    "Jumanji": [1, 0, 5, 4, 2],
    "Grumpier Old Men": [4, 5, 0, 0, 1],
    "Heat": [0, 4, 5, 1, 5],
    "Sabrina": [5, 0, 0, 4, 1]
}, index=["User1", "User2", "User3", "User4", "User5"])

similarity = pd.DataFrame(cosine_similarity(ratings),
                          index=ratings.index, columns=ratings.index)
user = "User1"
recommendations = {}

for movie in ratings.columns[ratings.loc[user] == 0]:
    mask = ratings[movie] > 0
    weights = similarity.loc[user, mask]
    if weights.sum() > 0:
        recommendations[movie] = (weights * ratings.loc[mask, movie]).sum() / weights.sum()

print("Recommendations for", user)
for movie, score in sorted(recommendations.items(), key=lambda x: x[1], reverse=True):
    print(f"{movie}: {score:.2f}")

#%% EXPERIMENT 16: ITEM-BASED COLLABORATIVE FILTERING
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.DataFrame({
    "Toy Story": [5, 4, 1, 0, 0],
    "Jumanji": [1, 0, 5, 4, 2],
    "Grumpier Old Men": [4, 5, 0, 0, 1],
    "Heat": [0, 4, 5, 1, 5],
    "Sabrina": [5, 0, 0, 4, 1]
}, index=["User1", "User2", "User3", "User4", "User5"])

similarity = pd.DataFrame(cosine_similarity(ratings.T),
                          index=ratings.columns, columns=ratings.columns)
user = "User1"
rated = ratings.columns[ratings.loc[user] > 0]
unrated = ratings.columns[ratings.loc[user] == 0]
recommendations = {}

for movie in unrated:
    weights = similarity.loc[movie, rated]
    scores = ratings.loc[user, rated]
    if weights.sum() > 0:
        recommendations[movie] = (weights * scores).sum() / weights.sum()

print("Recommendations for", user)
for movie, score in sorted(recommendations.items(), key=lambda x: x[1], reverse=True):
    print(f"{movie}: {score:.2f}")


#%% EXPERIMENT 17: LINEAR SVM
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

iris = load_iris()
mask = iris.target != 2
X, y = iris.data[mask, :2], iris.target[mask]

svm = SVC(kernel="linear", C=1).fit(X, y)
plt.figure(figsize=(8, 6))
DecisionBoundaryDisplay.from_estimator(svm, X, response_method="decision_function",
                                       plot_method="contour", levels=[-1, 0, 1],
                                       colors="k", linestyles=["--", "-", "--"],
                                       ax=plt.gca())
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
            s=150, facecolors="none", edgecolors="k", label="Support Vectors")
plt.title("Linear SVM Decision Boundary")
plt.legend()
plt.show()

#%% EXPERIMENT 18: RBF AND POLYNOMIAL KERNEL SVM
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

for kernel in ["rbf", "poly"]:
    model = SVC(kernel=kernel, degree=3).fit(X_train_s, y_train)
    print(f"{kernel.upper()} SVM Accuracy: "
          f"{accuracy_score(y_test, model.predict(X_test_s)):.4f}")


#%% EXPERIMENT 19: SVM HYPERPARAMETER TUNING AND ROC CURVE
# Reconstructed as a complete program because the PDF shows only its continuation.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

params = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]}
grid = GridSearchCV(SVC(kernel="rbf"), params, cv=5, scoring="accuracy")
grid.fit(X_train_s, y_train)
best_svm = grid.best_estimator_

print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)
print("Test Accuracy:", best_svm.score(X_test_s, y_test))

scores = best_svm.decision_function(X_test_s)
fpr, tpr, _ = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)
print("AUC:", round(roc_auc, 4))

plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned SVM")
plt.legend()
plt.grid(True)
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
