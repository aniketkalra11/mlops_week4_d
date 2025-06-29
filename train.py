import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import joblib
from sklearn.datasets import load_iris

# --------- Load your custom Iris-like dataset ---------
# Replace 'your_dataset.csv' with your actual dataset path
# It should have features as columns and the label column named 'target'
# data = pd.read_csv('your_custom_iris.csv')  # Replace with your dataset
iris = load_iris()
X = iris.data
y = iris.target

# --------- Train/test split ---------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------- Simulate training across multiple epochs ---------
epochs = 10
metrics = []

for epoch in range(1, epochs + 1):
    clf = RandomForestClassifier(
        random_state=42 + epoch,
        n_estimators=10 + epoch * 5,  # increasing complexity per epoch
        warm_start=False  # no incremental training
    )
    clf.fit(X_train, y_train)

    # Train predictions
    y_train_pred = clf.predict(X_train)
    y_train_proba = clf.predict_proba(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_loss = log_loss(y_train, y_train_proba)

    # Test predictions
    y_test_pred = clf.predict(X_test)
    y_test_proba = clf.predict_proba(X_test)
    val_acc = accuracy_score(y_test, y_test_pred)
    val_loss = log_loss(y_test, y_test_proba)

    metrics.append({
        'epoch': epoch,
        'accuracy': train_acc,
        'loss': train_loss,
        'val_accuracy': val_acc,
        'val_loss': val_loss
    })

# --------- Save final model ---------
joblib.dump(clf, 'iris_randomforest_model.joblib')
print("✅ Model saved to iris_randomforest_model.joblib")

# --------- Save metrics ---------
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('metrics.csv', index=False)
print("📊 Metrics saved to metrics.csv")
