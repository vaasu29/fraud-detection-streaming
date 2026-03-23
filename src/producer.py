import json
import time
import random
import pandas as pd
import numpy as np
from kafka import KafkaProducer
from src.preprocess import load_data, preprocess

def create_producer():
    return KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8')
    )

def simulate_transactions(producer, df, speed=1.0):
    """
    Stream real rows from creditcard.csv to Kafka topic.
    speed: seconds between messages (default 1.0)
    """
    print(f"Starting transaction stream to Kafka topic: transactions")
    print(f"Speed: 1 transaction every {speed}s — Ctrl+C to stop\n")

    sent = 0
    fraud_sent = 0

    for idx, row in df.iterrows():
        transaction = row.to_dict()

        # Add metadata
        transaction['transaction_id'] = f"txn_{idx}"
        transaction['timestamp'] = time.time()

        # Convert numpy types to native Python for JSON serialization
        transaction = {k: float(v) if isinstance(v, (np.floating, np.integer))
                      else v for k, v in transaction.items()}

        is_fraud = int(transaction.get('Class', 0))

        producer.send(
            topic='transactions',
            key=transaction['transaction_id'],
            value=transaction
        )
        producer.flush()

        sent += 1
        if is_fraud:
            fraud_sent += 1
            print(f"[FRAUD]  txn_{idx} sent | Total: {sent} | Frauds: {fraud_sent}")
        else:
            if sent % 100 == 0:
                print(f"[LEGIT]  txn_{idx} sent | Total: {sent} | Frauds: {fraud_sent}")

        time.sleep(speed)

if __name__ == "__main__":
    # Load real data
    df = load_data("creditcard.csv")
    X, y, scaler = preprocess(df)

    # Reconstruct full df with Class for streaming
    full_df = X.copy()
    full_df['Class'] = y.values

    producer = create_producer()

    try:
        simulate_transactions(producer, full_df, speed=0.1)
    except KeyboardInterrupt:
        print("\nProducer stopped.")
    finally:
        producer.close()
        print("Kafka producer closed.")
