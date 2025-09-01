# ========================================
# All Classification Algorithms in One Code
# ========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, Binarizer

# Models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# -----------------------
# Dataset
# -----------------------
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                           n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Models Dictionary
# -----------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN Classification": KNeighborsClassifier(n_neighbors=5),
    "SVM Classification": SVC(kernel="rbf", probability=True),
    "Gaussian Naive Bayes": GaussianNB(),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Bernoulli Naive Bayes": BernoulliNB(),
    "Decision Tree Classification": DecisionTreeClassifier(random_state=42),
    "Random Forest Classification": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting Classification": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42),
    "CatBoost": CatBoostClassifier(n_estimators=100, random_state=42, verbose=0),
    "SGD Classifier": SGDClassifier(loss="hinge", max_iter=1000, random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(50,30), max_iter=1000, random_state=42)
}

# -----------------------
# Evaluation Function
# -----------------------
def evaluate_model(y_true, y_pred, name):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }

# -----------------------
# Training & Evaluation
# -----------------------
results = []

for name, model in models.items():
    if name == "Multinomial Naive Bayes":
        scaler = MinMaxScaler()
        X_train_mod = scaler.fit_transform(X_train)
        X_test_mod = scaler.transform(X_test)
    elif name == "Bernoulli Naive Bayes":
        binarizer = Binarizer(threshold=0.0)
        X_train_mod = binarizer.fit_transform(X_train)
        X_test_mod = binarizer.transform(X_test)
    else:
        X_train_mod, X_test_mod = X_train, X_test

    model.fit(X_train_mod, y_train)
    y_pred = model.predict(X_test_mod)
    results.append(evaluate_model(y_test, y_pred, name))

# -----------------------
# Results Table
# -----------------------
results_df = pd.DataFrame(results)
print("\nClassification Model Comparison:\n")
print(results_df.sort_values(by="Accuracy", ascending=False))

# -----------------------
# Visualization
# -----------------------
plt.figure(figsize=(12,6))
sns.barplot(data=results_df.sort_values(by="Accuracy", ascending=False),
            x="Accuracy", y="Model", palette="viridis")
plt.title("Model Comparison (Accuracy)")
plt.show()
