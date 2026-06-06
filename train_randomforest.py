import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
    precision_recall_curve
)

# 1. Load dataset
# Make sure new_intrusion_dataset.csv is in the same folder as this script
df = pd.read_csv("new_intrusion_dataset.csv")

# If your file has no header row, add column names:
# df = pd.read_csv("new_intrusion_dataset.csv", header=None)
# df.columns = ["duration", "protocol_type", "service", "src_bytes",
#               "dst_bytes", "count", "score", "class"]

# 2. Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

# 3. Encode categorical columns and target
cat_cols = X.select_dtypes(include=["object"]).columns

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le  # keep encoder for inference later

target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)  # "intrusion"/"normal" -> 0/1

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Define and train RandomForest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 6. Save model and encoders with your desired name
print("Current working directory:", os.getcwd())

save_dir = os.getcwd()  # change if you want a different folder
model_path = os.path.join(save_dir, "randomforestintrusion_model.joblib")
encoders_path = os.path.join(save_dir, "randomforestintrusion_encoders.joblib")
target_path = os.path.join(save_dir, "randomforestintrusion_target.joblib")

joblib.dump(model, model_path)
joblib.dump(encoders, encoders_path)
joblib.dump(target_le, target_path)

print("Model saved to:", model_path)
print("Feature encoders saved to:", encoders_path)
print("Target encoder saved to:", target_path)

# 7. Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability of class "intrusion" (encoded)

# 8. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_le.classes_
)
disp.plot(values_format="d", cmap="Blues")
plt.title("Confusion Matrix - Intrusion Detection")
plt.tight_layout()
plt.show()

# 9. F1 score and classification report
f1 = f1_score(y_test, y_pred, average="binary")
print("F1-score (binary):", f1)
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, target_names=target_le.classes_))

# 10. F1 curve vs threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Compute F1 for each threshold (skip precision[0], recall[0])
f1_scores = []
for p, r in zip(precision[1:], recall[1:]):
    if p + r == 0:
        f1_scores.append(0.0)
    else:
        f1_scores.append(2 * p * r / (p + r))

used_thresholds = thresholds  # same length as precision[1:], recall[1:]

plt.figure()
plt.plot(used_thresholds, f1_scores, marker=".")
plt.xlabel("Threshold")
plt.ylabel("F1-score")
plt.title("F1-score vs Threshold (F1 Curve)")
plt.grid(True)
plt.tight_layout()
plt.show()
