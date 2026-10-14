
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

data = pd.read_csv("dataset/heart.csv")

print("Dataset Shape:", data.shape)
print("\nFirst Five Records:")
print(data.head())

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/decision_tree_model.pkl")

print("\nDecision Tree model saved successfully.")
print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Heart Disease Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

plt.figure(figsize=(22, 12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree for Heart Disease Prediction")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=200)
plt.show()
