# 🏗️ Architecture — Real-Time Fraud Detection with Streaming ML

## Overview

This document describes the end-to-end architecture of the real-time fraud detection pipeline, covering data ingestion, stream processing, model inference, storage, and serving layers.

---

## 📊 Pipeline Flow
```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCE                             │
│          creditcard.csv (284,807 transactions)              │
│          Kaggle — Credit Card Fraud Detection               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  KAFKA PRODUCER                             │
│               src/producer.py                               │
│                                                             │
│  • Streams real transactions row-by-row                     │
│  • Configurable speed (default: 0.1s per message)           │
│  • Serializes each row as JSON                              │
│  • Key: transaction_id, Value: feature JSON                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  APACHE KAFKA                               │
│              Topic: transactions                            │
│         (Docker — confluentinc/cp-kafka:7.4.0)             │
│                                                             │
│  • 1 partition, replication factor 1                        │
│  • Auto topic creation enabled                              │
│  • Zookeeper for cluster coordination                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            PYSPARK STRUCTURED STREAMING                     │
│           src/streaming_consumer.py                         │
│                                                             │
│  • Reads Kafka topic as unbounded DataFrame                 │
│  • Parses JSON using defined StructType schema              │
│  • Applies XGBoost UDF for real-time inference              │
│  • Micro-batch trigger: every 5 seconds                     │
│  • Adds fraud_probability, prediction, alert columns        │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────────────┐
│   CONSOLE SINK       │  │         DELTA LAKE SINK           │
│   (Monitoring)       │  │    delta_output/predictions       │
│                      │  │                                   │
│ • Prints batches     │  │ • ACID-compliant writes           │
│ • Shows FRAUD alerts │  │ • Append-only streaming mode      │
│ • Real-time view     │  │ • Queryable via FastAPI           │
└──────────────────────┘  └───────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   FASTAPI ENDPOINT                          │
│                    api/app.py                               │
│                                                             │
│  GET  /              — Health check                         │
│  GET  /health        — Model status                         │
│  POST /predict       — Single transaction inference         │
│  GET  /predictions/recent — Query Delta Lake results        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Model Layer
```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
│                     src/train.py                            │
│                                                             │
│  Raw Data (creditcard.csv)                                  │
│       │                                                     │
│       ▼                                                     │
│  Preprocessing (src/preprocess.py)                          │
│  • StandardScaler on Amount + Time                          │
│  • V1-V28 already PCA-transformed                           │
│       │                                                     │
│       ▼                                                     │
│  Train/Test Split (80/20, stratified)                       │
│       │                                                     │
│       ▼                                                     │
│  SMOTE Oversampling                                         │
│  • Balances 0.17% fraud rate → 50/50                        │
│  • Synthetic minority oversampling                          │
│       │                                                     │
│       ▼                                                     │
│  XGBoost Classifier                                         │
│  • n_estimators=100, max_depth=6                            │
│  • learning_rate=0.1                                        │
│       │                                                     │
│       ▼                                                     │
│  MLflow Logging                                             │
│  • Experiment: fraud-detection                              │
│  • Logs: params, metrics, model artifact                    │
│  • Model Registry: fraud-detection-xgboost v1              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.69% |
| **Recall (Fraud)** | **86.73%** |
| ROC-AUC | 97.60% |
| Precision | 34.55% |
| F1-Score | 49.42% |

> Recall is the primary metric — missing a fraud costs more than a false alarm.

---

## 🔄 Streaming Inference Detail
```
Kafka Message (JSON)
        │
        ▼
PySpark parses JSON → DataFrame row
        │
        ▼
XGBoost UDF called per row
  • Loads model from models/fraud_model.pkl
  • Extracts V1-V28 + Amount_scaled + Time_scaled
  • Returns fraud_probability (float)
        │
        ▼
Threshold applied (≥ 0.5 → FRAUD)
        │
        ▼
Risk level assigned:
  • prob ≥ 0.8  → HIGH
  • prob ≥ 0.5  → MEDIUM
  • prob < 0.5  → LOW
        │
        ▼
Written to Delta Lake + Console
```

---

## 🗂️ Component Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| Message Broker | Apache Kafka 7.4.0 | Real-time transaction ingestion |
| Stream Processor | PySpark 3.5.1 | Micro-batch inference engine |
| ML Model | XGBoost 3.2.0 + SMOTE | Fraud classification |
| Experiment Tracking | MLflow 3.10.1 | Model versioning + metrics |
| Storage | Delta Lake 3.2.0 | ACID prediction persistence |
| API Layer | FastAPI 0.135.1 | REST inference + query endpoint |
| Containerization | Docker + Compose | Kafka + Zookeeper deployment |
| Coordination | Apache Zookeeper | Kafka cluster management |

---

## 🚀 Scalability Path

| Current (Local) | Production (Databricks) |
|-----------------|------------------------|
| 1 Kafka partition | N partitions = N parallel consumers |
| Local Spark (2g RAM) | Databricks cluster (auto-scaling) |
| Python UDF (row-by-row) | Pandas UDF (vectorized, 10-100x faster) |
| Local Delta Lake | Databricks Delta Lake (managed) |
| File-based MLflow | Databricks MLflow (managed registry) |
| Manual retraining | Automated retraining on drift detection |

---

## 👤 Author
**Vaasu Chandra** — [GitHub](https://github.com/vaasu29) · [LinkedIn](https://linkedin.com/in/vaasuchandra)
