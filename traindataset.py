import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ------------------------------------
# File paths
# ------------------------------------
TRAIN_CSV = "new_intrusion_dataset.csv"
TEST_CSV = "new_intrusion_dataset.csv"
MODEL_FILE = "knn_intrusion_model.joblib"

# ------------------------------------
# Load dataset
# ------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {path} with shape {df.shape}")
    return df

# ------------------------------------
# Prepare X and y
# ------------------------------------
def prepare_xy(df, label_col):

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' missing!")

    X = df.drop(columns=[label_col])
    y = df[label_col].astype(str)

    # Automatically detect numeric + categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    print("\nDetected Numeric Columns:", num_cols)
    print("Detected Categorical Columns:", cat_cols)

    return X, y, num_cols, cat_cols


# ------------------------------------
# Build pipeline
# ------------------------------------
def build_pipeline(num_cols, cat_cols):

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    model = KNeighborsClassifier(n_neighbors=5)

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


# ------------------------------------
# Confusion Matrix Plot
# ------------------------------------
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ------------------------------------
# Precision–Recall Curve
# ------------------------------------
def plot_precision_recall(y_true, y_scores, labels):
    plt.figure(figsize=(7, 5))
    for i, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(
            (y_true == i).astype(int), y_scores[:, i]
        )
        plt.plot(recall, precision, label=str(label))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------
# Main
# ------------------------------------
def main():

    df_train = load_data(TRAIN_CSV)
    df_test = load_data(TEST_CSV) if os.path.exists(TEST_CSV) else None

    label_col = "class"

    X_train, y_train, num_cols, cat_cols = prepare_xy(df_train, label_col)

    print("\nTraining columns:", X_train.columns.tolist())

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    pipeline = build_pipeline(num_cols, cat_cols)

    print("\nTraining model...")
    pipeline.fit(X_train, y_train_enc)
    print("Model trained successfully ✔")

    # ---------------- Evaluation ----------------
    if df_test is not None and label_col in df_test.columns:
        print("\nEvaluating on Test Data...")
        X_test, y_test, _, _ = prepare_xy(df_test, label_col)
        y_test_enc = le.transform(y_test)

        y_pred = pipeline.predict(X_test)
        y_scores = pipeline.predict_proba(X_test)

        print("\nAccuracy:", accuracy_score(y_test_enc, y_pred))
        print("\nClassification Report:\n", classification_report(y_test_enc, y_pred, target_names=le.classes_))

        plot_confusion_matrix(y_test_enc, y_pred, le.classes_)
        plot_precision_recall(y_test_enc, y_scores, le.classes_)

    else:
        print("\n⚠ No test data found — evaluating on training data...")
        y_pred = pipeline.predict(X_train)
        y_scores = pipeline.predict_proba(X_train)

        print("\nClassification Report:\n", classification_report(y_train_enc, y_pred, target_names=le.classes_))

        plot_confusion_matrix(y_train_enc, y_pred, le.classes_)
        plot_precision_recall(y_train_enc, y_scores, le.classes_)

    # Save model
    print(f"\nSaving model to {MODEL_FILE} ...")
    joblib.dump({
        "pipeline": pipeline,
        "label_encoder": le,
        "numeric_features": num_cols,
        "categorical_features": cat_cols
    }, MODEL_FILE)

    print("✔ Model saved successfully!")


if __name__ == "__main__":
    main()
