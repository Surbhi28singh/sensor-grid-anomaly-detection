"""
🌟 Intelligent Sensor Grid Anomaly Detection

Hybrid Machine Learning + Autoencoder + Drift Detection
Single-file production-ready pipeline

Author: Surbhi Singh
"""

# -----------------------------
# 📦 IMPORTS
# -----------------------------
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# 📊 STEP 1: GENERATE SYNTHETIC SENSOR DATA
# -----------------------------
def generate_sensor_data(num_samples=5000, num_sensors=5, anomaly_ratio=0.05):
    """
    Generates synthetic sensor data for normal operation and anomalies.

    Normal readings: Gaussian distribution
    Anomalous readings: Uniform distribution
    """
    np.random.seed(42)

    # Normal sensor readings
    normal = np.random.normal(loc=50, scale=10, size=(int(num_samples*(1-anomaly_ratio)), num_sensors))
    
    # Anomalous sensor readings
    anomalies = np.random.uniform(low=5, high=100, size=(int(num_samples*anomaly_ratio), num_sensors))
    
    # Combine data and labels
    data = np.vstack((normal, anomalies))
    labels = np.hstack((np.zeros(len(normal)), np.ones(len(anomalies))))

    df = pd.DataFrame(data, columns=[f"sensor_{i+1}" for i in range(num_sensors)])
    df['anomaly'] = labels

    return df

# -----------------------------
# 🧼 STEP 2: PREPROCESS DATA
# -----------------------------
def preprocess_data(df):
    """
    Standardizes features and separates labels.
    """
    scaler = StandardScaler()
    X = df.drop("anomaly", axis=1)
    y = df["anomaly"]
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# -----------------------------
# 🧠 STEP 3: BUILD AUTOENCODER
# -----------------------------
def build_autoencoder(input_dim):
    """
    Constructs a simple fully connected autoencoder.
    """
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(64, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return autoencoder

# -----------------------------
# 🤖 STEP 4: TRAIN HYBRID MODEL
# -----------------------------
def train_hybrid_model(X_train, y_train, X_test, y_test):
    """
    Trains Random Forest + Autoencoder and performs ensemble prediction.
    """
    # --- Random Forest ---
    rf_model = RandomForestClassifier(n_estimators=120, random_state=42)
    rf_model.fit(X_train, y_train)

    # --- Autoencoder ---
    ae_model = build_autoencoder(X_train.shape[1])
    ae_model.fit(X_train, X_train, epochs=20, batch_size=32, verbose=0)

    # Determine threshold for anomaly detection based on reconstruction error
    reconstructed = ae_model.predict(X_train, verbose=0)
    reconstruction_error = np.mean(np.square(X_train - reconstructed), axis=1)
    threshold = np.percentile(reconstruction_error, 95)

    # Function to predict anomalies using autoencoder
    def autoencoder_predict(X):
        recon = ae_model.predict(X, verbose=0)
        error = np.mean(np.square(X - recon), axis=1)
        return (error > threshold).astype(int)

    # --- Ensemble prediction ---
    rf_pred = rf_model.predict(X_test)
    ae_pred = autoencoder_predict(X_test)
    ensemble_pred = ((rf_pred + ae_pred) > 0).astype(int)

    print("\n📌 HYBRID MODEL CLASSIFICATION REPORT:")
    print(classification_report(y_test, ensemble_pred))

    return rf_model, ae_model, threshold

# -----------------------------
# ⚠️ STEP 5: SIMPLE DRIFT DETECTION
# -----------------------------
def detect_drift(new_data, reference_mean, tolerance=0.2):
    """
    Detects data drift based on mean shift of new batch.
    """
    batch_mean = np.mean(new_data)
    drift_ratio = abs(batch_mean - reference_mean) / reference_mean
    return drift_ratio > tolerance

# -----------------------------
# 🚀 STEP 6: REAL-TIME PREDICTION
# -----------------------------
def predict_realtime(rf_model, ae_model, threshold, scaler, batch_data):
    """
    Predicts anomalies for new sensor data in real-time.
    Returns predictions and reconstruction errors.
    """
    batch_scaled = scaler.transform(batch_data)
    rf_pred = rf_model.predict(batch_scaled)
    recon = ae_model.predict(batch_scaled, verbose=0)
    recon_error = np.mean(np.square(batch_scaled - recon), axis=1)
    ae_pred = (recon_error > threshold).astype(int)
    final_pred = ((rf_pred + ae_pred) > 0).astype(int)
    return final_pred, recon_error

# -----------------------------
# 🏁 MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    print("\n🚀 Starting Intelligent Sensor Grid Anomaly Detection")

    # Generate synthetic sensor data
    df = generate_sensor_data()

    # Preprocess data
    X_scaled, y, scaler = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )

    # Train hybrid model
    rf_model, ae_model, threshold = train_hybrid_model(X_train, y_train, X_test, y_test)

    # Reference mean for drift detection
    reference_mean = np.mean(X_train)

    # Simulate a new batch
    new_batch = np.random.normal(55, 12, size=(20, 5))

    # Check for drift
    if detect_drift(new_batch, reference_mean):
        print("\n⚠️ DATA DRIFT DETECTED! Consider retraining the model.")
    else:
        print("\n✅ No data drift detected.")

    # Real-time anomaly prediction
    predictions, errors = predict_realtime(rf_model, ae_model, threshold, scaler, new_batch)
    print("\n🔍 Real-time anomaly predictions:")
    print(predictions)

