from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
import uuid
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ---------------- EMAIL SETTINGS ----------------
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_USER = os.getenv('EMAIL_USER', 'myprojectemails4u@gmail.com')
EMAIL_PASS = os.getenv('EMAIL_PASS', 'cyaoslrmrystypcm')
EMAIL_USE_SSL = os.getenv('EMAIL_USE_SSL', 'False').lower() in ('true', '1', 'yes')

app = Flask(__name__)
app.secret_key = "super_secret_key"

# ---------------- LOAD KNN MODEL ----------------
MODEL_PATH = "knn_intrusion_model.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model missing! Train and place knn_intrusion_model.joblib in root folder.")
model_data = joblib.load(MODEL_PATH)
pipeline = model_data["pipeline"]
label_encoder = model_data["label_encoder"]
NUMERIC_FEATURES = model_data.get("numeric_features", [])
CATEGORICAL_FEATURES = model_data.get("categorical_features", [])

# ---------------- LOAD RANDOM FOREST MODEL ----------------
RF_MODEL_PATH = "randomforestintrusion_model.joblib"
RF_ENCODERS_PATH = "randomforestintrusion_encoders.joblib"
RF_TARGET_PATH = "randomforestintrusion_target.joblib"

rf_model = None
rf_encoders = None
rf_target_encoder = None

if os.path.exists(RF_MODEL_PATH) and os.path.exists(RF_ENCODERS_PATH) and os.path.exists(RF_TARGET_PATH):
    rf_model = joblib.load(RF_MODEL_PATH)
    rf_encoders = joblib.load(RF_ENCODERS_PATH)
    rf_target_encoder = joblib.load(RF_TARGET_PATH)
    print("Random Forest model loaded successfully!")
else:
    print(" Random Forest model files missing.")

# Random Forest expected features
RF_FEATURES = ["duration", "protocol_type", "service", "src_bytes", "dst_bytes", "count", "serror_rate"]

# ---------------- MYSQL CONFIG ----------------
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'intrusiondetection_sdm',
    'port': 3308,
    'charset': 'utf8'
}

db_available = True
local_users = {}
local_prediction_logs = []


def get_db_connection():
    global db_available
    try:
        conn = mysql.connector.connect(**db_config)
        db_available = True
        return conn
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        db_available = False
        return None


# ---------------- CREATE LOG TABLE ----------------
def create_log_table():
    conn = get_db_connection()
    if conn is None:
        print("Skipping log table creation because the database is unavailable.")
        return

    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logtable (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_email VARCHAR(255),
                algorithm VARCHAR(50),
                duration FLOAT,
                protocol_type VARCHAR(20),
                service VARCHAR(50),
                src_bytes FLOAT,
                dst_bytes FLOAT,
                count FLOAT,
                serror_rate FLOAT,
                prediction VARCHAR(50),
                prediction_proba FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("Log table ready!")
    except Exception as e:
        print(f"Error creating log table: {e}")
        db_available = False


create_log_table()


# ---------------- SAVE PREDICTION LOG ----------------
def save_prediction_log(user_email, algorithm, input_data, prediction, proba):
    if not db_available:
        local_prediction_logs.append({
            'user_email': user_email,
            'algorithm': algorithm,
            'duration': input_data.get('duration', 0),
            'protocol_type': input_data.get('protocol_type', ''),
            'service': input_data.get('service', ''),
            'src_bytes': input_data.get('src_bytes', 0),
            'dst_bytes': input_data.get('dst_bytes', 0),
            'count': input_data.get('count', 0),
            'serror_rate': input_data.get('serror_rate', 0.5),
            'prediction': prediction,
            'prediction_proba': proba,
            'created_at': datetime.now()
        })
        return True

    conn = get_db_connection()
    if conn is None:
        local_prediction_logs.append({
            'user_email': user_email,
            'algorithm': algorithm,
            'duration': input_data.get('duration', 0),
            'protocol_type': input_data.get('protocol_type', ''),
            'service': input_data.get('service', ''),
            'src_bytes': input_data.get('src_bytes', 0),
            'dst_bytes': input_data.get('dst_bytes', 0),
            'count': input_data.get('count', 0),
            'serror_rate': input_data.get('serror_rate', 0.5),
            'prediction': prediction,
            'prediction_proba': proba,
            'created_at': datetime.now()
        })
        return True

    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO logtable (user_email, algorithm, duration, protocol_type, service, 
                                src_bytes, dst_bytes, count, serror_rate, prediction, prediction_proba)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_email, algorithm,
            input_data.get('duration', 0),
            input_data.get('protocol_type', ''),
            input_data.get('service', ''),
            input_data.get('src_bytes', 0),
            input_data.get('dst_bytes', 0),
            input_data.get('count', 0),
            input_data.get('serror_rate', 0.5),
            prediction, proba
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving log: {e}")
        return False


# ---------------- GET PREDICTION LOGS ----------------
def get_prediction_logs(user_email=None, limit=10):
    if not db_available:
        filtered = [log for log in local_prediction_logs if not user_email or log['user_email'] == user_email]
        return sorted(filtered, key=lambda x: x['created_at'], reverse=True)[:limit]

    conn = get_db_connection()
    if conn is None:
        filtered = [log for log in local_prediction_logs if not user_email or log['user_email'] == user_email]
        return sorted(filtered, key=lambda x: x['created_at'], reverse=True)[:limit]

    try:
        cursor = conn.cursor(dictionary=True)
        if user_email:
            cursor.execute("""
                SELECT * FROM logtable 
                WHERE user_email = %s 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (user_email, limit))
        else:
            cursor.execute("""
                SELECT * FROM logtable 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (limit,))
        logs = cursor.fetchall()
        cursor.close()
        conn.close()
        return logs
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return []


# ---------------- MAC ADDRESS ----------------
def get_mac_address():
    mac = ':'.join(['{:02x}'.format((uuid.getnode() >> ele) & 0xff)
                    for ele in range(0, 8 * 6, 8)][::-1])
    return mac


# ---------------- Random Forest Prediction Function ----------------
def rf_predict_intrusion(data_dict):
    if rf_model is None or rf_encoders is None or rf_target_encoder is None:
        return "Model not loaded", 0.0

    try:
        df = pd.DataFrame([data_dict])
        for col, le in rf_encoders.items():
            if col in df.columns:
                for v in df[col].unique():
                    if v not in le.classes_:
                        le.classes_ = np.append(le.classes_, v)
                df[col] = le.transform(df[col])

        pred_encoded = rf_model.predict(df)[0]
        pred_label = rf_target_encoder.inverse_transform([pred_encoded])[0]
        proba = rf_model.predict_proba(df)[0]
        pred_proba = proba[list(rf_target_encoder.classes_).index(pred_label)]

        return pred_label, float(pred_proba)
    except Exception as e:
        return f"Error: {e}", 0.0


# ---------------- EMAIL FUNCTION ----------------
def send_email(to_email, subject, message):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, "plain"))

    if not EMAIL_USER or not EMAIL_PASS:
        print("Email credentials are not configured.")
        return False

    email_methods = []
    if EMAIL_USE_SSL:
        email_methods.append(('ssl', EMAIL_PORT))
    else:
        email_methods.append(('tls', EMAIL_PORT))
        email_methods.append(('ssl', 465))

    for method, port in email_methods:
        try:
            if method == 'ssl':
                server = smtplib.SMTP_SSL(EMAIL_HOST, port, timeout=10)
            else:
                server = smtplib.SMTP(EMAIL_HOST, port, timeout=10)
                server.ehlo()
                server.starttls()
                server.ehlo()

            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
            server.quit()
            return True
        except smtplib.SMTPAuthenticationError as e:
            print(f"Email auth error ({method}): {e}")
            if method == 'ssl':
                break
        except Exception as e:
            print(f"Email sending error ({method}): {e}")
            if method == 'ssl':
                break
    return False


# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return redirect(url_for('home')) if 'user' in session else redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        mac = get_mac_address()
        conn = get_db_connection()
        if conn is None:
            if email in local_users:
                flash('Email already exists locally!', 'warning')
            else:
                local_users[email] = {
                    'name': name,
                    'email': email,
                    'password': password,
                    'mac_address': mac
                }
                flash('Registered locally. You can login until the database is available.', 'success')
            return render_template('register.html')

        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, email, password, mac_address) VALUES (%s, %s, %s, %s)",
                           (name, email, password, mac))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            flash('Email already exists!', 'warning')
        except Exception as e:
            print(f"Registration error: {e}")
            flash('Registration failed. Please try again.', 'danger')
        finally:
            try:
                cursor.close(); conn.close()
            except:
                pass
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        mac = get_mac_address()
        conn = get_db_connection()
        if conn is None:
            user = local_users.get(email)
            if user and user['password'] == password and user['mac_address'] == mac:
                session['user'] = user['email']
                flash('Logged in locally while the database is unavailable.', 'info')
                return redirect(url_for('home'))
            flash('Database is unavailable and local login failed.', 'danger')
            return render_template('login.html')

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE email=%s AND password=%s AND mac_address=%s",
                           (email, password, mac))
            user = cursor.fetchone()
        except Exception as e:
            print(f"Login error: {e}")
            user = None
        finally:
            try:
                cursor.close(); conn.close()
            except:
                pass

        if user:
            session['user'] = user['email']
            return redirect(url_for('home'))
        else:
            flash("Invalid login, unauthorized device, or database unavailable!", 'danger')
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html', user=session['user']) if 'user' in session else redirect(url_for('login'))


# -------- SINGLE PREDICTION --------
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None
    rf_prediction = None
    error_message = None
    prediction_logs = []

    if request.method == 'POST':
        model_type = request.form.get('model_type', 'knn')

        try:
            # Collect all input data
            input_data = {}
            input_data['duration'] = float(request.form.get('duration', 0))
            input_data['protocol_type'] = request.form.get('protocol_type', 'tcp')
            input_data['service'] = request.form.get('service', 'http')
            input_data['src_bytes'] = float(request.form.get('src_bytes', 0))
            input_data['dst_bytes'] = float(request.form.get('dst_bytes', 0))
            input_data['count'] = float(request.form.get('count', 0))
            input_data['serror_rate'] = float(request.form.get('serror_rate', 0.5))

            # Fill KNN features from form
            for col in CATEGORICAL_FEATURES:
                input_data[col] = request.form.get(col, '')
            for col in NUMERIC_FEATURES:
                val = request.form.get(col, 0)
                input_data[col] = float(val) if val else 0.0

            if model_type == 'knn':
                # KNN Prediction
                df_input = pd.DataFrame([input_data])
                pred_encoded = pipeline.predict(df_input)[0]
                pred_label = label_encoder.inverse_transform([pred_encoded])[0]
                prediction = "Normal" if pred_label.lower() == "normal" else "Intrusion"
                proba = 1.0  # KNN doesn't provide probability by default

                # Save to log
                save_prediction_log(session['user'], 'KNN', input_data, prediction, proba)

            elif model_type == 'rf':
                # Random Forest Prediction
                rf_pred_label, rf_proba = rf_predict_intrusion(input_data)
                rf_prediction = f"{rf_pred_label} ({rf_proba:.2%})"
                proba = rf_proba

                # Save to log
                save_prediction_log(session['user'], 'Random Forest', input_data, rf_pred_label, proba)

            # Send email
            user_email = session['user']
            subject = "Intrusion Detection Result"
            message = f"Your intrusion detection prediction result is:\n\n"
            if prediction:
                message += f"KNN Result: {prediction}\n"
            if rf_prediction:
                message += f"Random Forest Result: {rf_prediction}\n"

            email_sent = send_email(user_email, subject, message)
            if email_sent:
                flash("📩 Email sent successfully!", "success")
            else:
                flash("⚠️ Prediction done, but email sending failed! Check SMTP credentials or Gmail app password.", "danger")

        except Exception as e:
            error_message = f"Prediction error: {e}"

    # Get user's prediction logs
    prediction_logs = get_prediction_logs(session.get('user'))

    return render_template('prediction.html',
                           numeric_fields=NUMERIC_FEATURES,
                           categorical_fields=CATEGORICAL_FEATURES,
                           prediction=prediction,
                           rf_prediction=rf_prediction,
                           error_message=error_message,
                           rf_model_available=rf_model is not None,
                           prediction_logs=prediction_logs)


# -------- CSV FILE PREDICTION --------
@app.route('/predict_file', methods=['GET', 'POST'])
def predict_file():
    if 'user' not in session:
        return redirect(url_for('login'))
    prediction_results = None
    error_message = None
    if request.method == 'POST':
        try:
            file = request.files.get('csv_file')
            if not file:
                error_message = "No file uploaded!"
            else:
                df_input = pd.read_csv(file)
                preds_encoded = pipeline.predict(df_input)
                preds_labels = label_encoder.inverse_transform(preds_encoded)
                df_input['Prediction'] = [
                    "Normal" if p.lower().startswith("normal") else "Intrusion"
                    for p in preds_labels
                ]
                prediction_results = df_input.head(20).to_html(
                    classes='table table-bordered table-striped',
                    index=False
                )
        except Exception as e:
            error_message = f"Error processing file: {e}"
    return render_template('predict_file.html',
                           prediction_results=prediction_results,
                           error_message=error_message)


@app.route('/contact')
def contact():
    return render_template('contact.html', user=session['user']) if 'user' in session else redirect(url_for('login'))


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
