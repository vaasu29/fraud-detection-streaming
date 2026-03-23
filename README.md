# 🔍 Real-Time Fraud Detection with Streaming ML

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-3.5.1-E25A1C?style=flat&logo=apachespark&logoColor=white)
![Kafka](https://img.shields.io/badge/Kafka-7.4.0-231F20?style=flat&logo=apachekafka&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.10.1-0194E2?style=flat&logo=mlflow&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-3.2.0-00ADD8?style=flat)
![Docker](https://img.shields.io/badge/Docker-28.3.3-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

An end-to-end streaming ML pipeline that ingests live financial transactions via **Apache Kafka**, runs real-time fraud detection using a **XGBoost** model inside **PySpark Structured Streaming**, persists predictions to **Delta Lake**, and serves results via a **FastAPI** REST endpoint — all tracked with **MLflow**.

---

## 🏗️ Architecture
```
creditcard.csv (284K transactions)
        ↓
  Kafka Producer
  (simulates live transaction stream)
        ↓
  Kafka Topic: transactions
        ↓
  PySpark Structured Streaming
  (feature engineering + XGBoost UDF)
        ↓
  ┌─────────────────┬──────────────────┐
  │  Console Sink   │  Delta Lake Sink │
  │  (monitoring)   │  (persistence)   │
  └─────────────────┴──────────────────┘
        ↓
  FastAPI REST Endpoint
  (real-time inference + prediction query)
        ↓
  MLflow Model Registry
  (experiment tracking + model versioning)
```

---

## ✨ Features

- **Real-time ingestion** — Kafka producer streams 284K transactions at configurable speed
- **Streaming inference** — PySpark Structured Streaming runs XGBoost predictions every 5 seconds
- **Class imbalance handling** — SMOTE balances 0.17% fraud rate before training
- **Delta Lake persistence** — All predictions stored with ACID guarantees
- **MLflow tracking** — Every training run logged with params, metrics, and model artifacts
- **FastAPI serving** — REST endpoint for single transaction inference with risk scoring
- **Docker** — Kafka + Zookeeper fully containerized

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.69% |
| Recall (Fraud) | 86.73% |
| ROC-AUC | 97.60% |
| Precision | 34.55% |

> **Recall is the key metric** — missing a fraud is more costly than a false alarm. 86.73% recall on 0.17% imbalanced data with SMOTE is production-grade.

---

## 🗂️ Project Structure
```
fraud-detection-streaming/
├── docker-compose.yml          # Kafka + Zookeeper
├── requirements.txt
├── src/
│   ├── preprocess.py           # Feature engineering + scaler
│   ├── train.py                # XGBoost training + MLflow logging
│   ├── producer.py             # Kafka producer (simulates transactions)
│   └── streaming_consumer.py  # PySpark Structured Streaming + predictions
├── api/
│   └── app.py                  # FastAPI inference endpoint
├── delta_output/               # Delta Lake predictions (gitignored)
├── models/                     # Saved model artifacts (gitignored)
└── mlruns/                     # MLflow experiments (gitignored)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Java JDK 11
- Docker Desktop

### 1. Clone & Install
```bash
git clone https://github.com/vaasu29/fraud-detection-streaming.git
cd fraud-detection-streaming
pip install -r requirements.txt
```

### 2. Download Dataset
Download [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place `creditcard.csv` in the project root.

### 3. Start Kafka
```bash
docker compose up -d
docker exec kafka kafka-topics --create \
  --topic transactions \
  --bootstrap-server localhost:9092 \
  --partitions 1 --replication-factor 1
```

### 4. Train Model
```bash
python3 -m src.train
```

### 5. Start All Components (3 terminals)
```bash
# Terminal 1 — FastAPI
python3 -m uvicorn api.app:app --reload --port 8000

# Terminal 2 — PySpark Streaming Consumer
python3 -m src.streaming_consumer

# Terminal 3 — Kafka Producer
python3 -m src.producer
```

### 6. Test the API
```bash
curl http://127.0.0.1:8000/health
```
Swagger UI: **http://127.0.0.1:8000/docs**

---

## 🔬 MLflow Dashboard
```bash
python3 -m mlflow ui
```
Open **http://127.0.0.1:5000**

---

## 🛣️ Roadmap
- [x] Kafka producer simulating live transactions
- [x] PySpark Structured Streaming consumer
- [x] XGBoost model with SMOTE
- [x] MLflow experiment tracking + Model Registry
- [x] Delta Lake prediction persistence
- [x] FastAPI REST inference endpoint
- [x] Docker containerization
- [ ] CI/CD with GitHub Actions
- [ ] Cloud deployment (AWS/Databricks)
- [ ] Grafana dashboard for prediction monitoring

---

## 👤 Author
**Vaasu Chandra**
[![GitHub](https://img.shields.io/badge/GitHub-vaasu29-black?style=flat&logo=github)](https://github.com/vaasu29)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-vaasuchandra-blue?style=flat&logo=linkedin)](https://linkedin.com/in/vaasuchandra)
