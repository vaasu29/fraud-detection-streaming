import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(path="creditcard.csv"):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    return df

def preprocess(df):
    # Normalize Amount and Time — V1-V28 are already PCA transformed
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])

    # Drop original Amount and Time
    df = df.drop(columns=['Amount', 'Time'])

    # Features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    return X, y, scaler

def save_scaler(scaler, path="models/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")

def load_scaler(path="models/scaler.pkl"):
    return joblib.load(path)

if __name__ == "__main__":
    df = load_data("creditcard.csv")
    X, y, scaler = preprocess(df)
    save_scaler(scaler)
    print("Preprocessing complete!")
