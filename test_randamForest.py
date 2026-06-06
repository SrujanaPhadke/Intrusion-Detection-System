import numpy as np
import pandas as pd
import joblib

# 1. Load model and encoders
model = joblib.load("randomforestintrusion_model.joblib")
encoders = joblib.load("randomforestintrusion_encoders.joblib")
target_le = joblib.load("randomforestintrusion_target.joblib")

# 2. Function to build a single sample and predict
def predict_intrusion(
    duration,
    protocol_type,   # "tcp", "udp", "icmp"
    service,         # "http", "ftp", "dns", "ssh", "smtp", ...
    src_bytes,
    dst_bytes,
    count,
    serror_rate      # this must match the training column name
):
    # IMPORTANT: column names and order must match training
    data = {
        "duration": [duration],
        "protocol_type": [protocol_type],
        "service": [service],
        "src_bytes": [src_bytes],
        "dst_bytes": [dst_bytes],
        "count": [count],
        "serror_rate": [serror_rate],
    }
    df = pd.DataFrame(data)

    # Apply the same LabelEncoders to categorical columns
    for col, le in encoders.items():
        # Handle unseen labels
        for v in df[col].unique():
            if v not in le.classes_:
                le.classes_ = np.append(le.classes_, v)
        df[col] = le.transform(df[col])

    # 3. Predict
    pred_encoded = model.predict(df)[0]
    pred_label = target_le.inverse_transform([pred_encoded])[0]

    proba = model.predict_proba(df)[0]
    classes = target_le.classes_
    pred_index = list(classes).index(pred_label)
    pred_proba = proba[pred_index]

    return pred_label, float(pred_proba), classes, proba

# 4. Example usage
if __name__ == "__main__":
    label, prob, classes, all_proba = predict_intrusion(
        duration=100,
        protocol_type="tcp",
        service="http",
        src_bytes=5000,
        dst_bytes=10000,
        count=10,
        serror_rate=0.5    # use the same kind of value as in training
    )
    print("Classes order:", classes)
    print("Raw probabilities:", all_proba)
    print(f"Predicted class: {label}, probability: {prob:.4f}")
