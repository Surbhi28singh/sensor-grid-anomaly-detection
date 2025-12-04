Intelligent Sensor Grid Anomaly Detection
Hybrid ML + Autoencoder • Real-Time Streaming • Edge + Cloud Ready
<p align="center"> <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/> <img src="https://img.shields.io/badge/Machine%20Learning-Enabled-blue?style=for-the-badge"/> <img src="https://img.shields.io/badge/Deep%20Learning-Autoencoder-purple?style=for-the-badge"/> <img src="https://img.shields.io/badge/Real%20Time-Yes-orange?style=for-the-badge"/> </p>

A modern anomaly detection system for grid-based sensor networks using Machine Learning, Deep Learning, and Real-Time Streaming.
Designed as a Final-Year Major Project (300 Marks) with professional-grade architecture.

✨ Key Features

⚡ Hybrid Model: Random Forest + Autoencoder

📡 Real-Time Sensor Monitoring (Kafka/MQTT Ready)

🧠 Explainable AI: SHAP-based insights

🧭 Drift Detection for evolving sensor behavior

📊 Rich Dashboard: Live charts, anomalies, summaries

☁️ Edge + Cloud Deployment compatible

📱 Alerts: Email / SMS / WhatsApp

🎯 Project Overview

Sensor grids generate continuous data that may contain noise, faults, missing values, or attack patterns.
This system detects anomalies instantly, provides explanations, and supports scalable deployment.

🏗️ System Architecture
Sensors → Edge Module → Kafka/MQTT → ML Engine (RF + Autoencoder)
        → Cloud/MongoDB → Streamlit Dashboard → Alerts




🛠️ Tech Stack

Machine Learning:
Random Forest • Autoencoder • Isolation Forest

Languages & Frameworks:
Python • TensorFlow • Scikit-Learn • NumPy • Pandas

Pipeline & Messaging:
Kafka • MQTT

Storage:
MongoDB • AWS S3

Visualization:
Streamlit • Plotly

📊 Results

🔍 High anomaly detection accuracy

📉 Low false-positive rate

⚡ Millisecond-level edge inference

🧠 SHAP plots for model interpretability

▶️ How to Run
pip install -r requirements.txt

# Generate sensor data
python src/data_generator.py

# Train the models
python src/model_training.py

# Launch dashboard
streamlit run dashboard/app.py

🚀 Future Improvements

LoRaWAN long-range sensor support

Transformer-based anomaly detection

Blockchain for sensor-log integrity

Mobile app with real-time visualization

👤 Developer

Surbhi Singh
B.Tech CSE | AI/ML • IoT • Data Science
⭐ If you like this project, consider giving it a star!
