import joblib
import pandas as pd
import numpy as np

model_data = joblib.load("knn_intrusion_model.joblib")
pipeline = model_data["pipeline"]
le = model_data["label_encoder"]

# Example single sample as dict (fill required features or rely on defaults as earlier)
sample = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 491,
    "dst_bytes": 0,
    # other numeric features will be added automatically if you used the earlier route
}

# Convert to DataFrame and ensure feature columns exist (use saved feature lists if needed)
expected = model_data.get("numeric_features", []) + model_data.get("categorical_features", [])
df = pd.DataFrame([sample])
for c in expected:
    if c not in df.columns:
        df[c] = 0 if c in model_data.get("numeric_features", []) else "unknown"
df = df[expected]

# Predicted encoded label and readable label
pred_enc = pipeline.predict(df)
pred_label = le.inverse_transform(pred_enc)[0]

# Probability (KNN supports predict_proba)
if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
    probs = pipeline.predict_proba(df)[0]
    class_probs = dict(zip(le.classes_, probs))
else:
    class_probs = None

is_intrusion = (pred_label != "normal")  # adjust if your normal label differs

print("Predicted label:", pred_label)
print("Is intrusion?:", is_intrusion)
print("Class probabilities:", class_probs)
