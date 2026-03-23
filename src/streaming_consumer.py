import os
import json
import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (from_json, col, udf,
                                   current_timestamp, when)
from pyspark.sql.types import (StructType, StructField, StringType,
                                DoubleType, TimestampType)
from delta import configure_spark_with_delta_pip

# ── Spark Session with Delta Lake + Kafka ──────────────────────────
def create_spark_session():
    builder = SparkSession.builder \
        .appName("FraudDetectionStreaming") \
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.streaming.checkpointLocation", "checkpoints/") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4")

    # Let delta pip handle its own JARs, then add Kafka separately
    spark = configure_spark_with_delta_pip(
        builder,
        extra_packages=["org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"]
    ).getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("Spark session created with Delta Lake + Kafka support")
    return spark

# ── Kafka Message Schema ───────────────────────────────────────────
def get_schema():
    fields = [StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)]
    fields += [
        StructField("Amount_scaled",  DoubleType(), True),
        StructField("Time_scaled",    DoubleType(), True),
        StructField("Class",          DoubleType(), True),
        StructField("transaction_id", StringType(), True),
        StructField("timestamp",      DoubleType(), True),
    ]
    from pyspark.sql.types import StructType
    return StructType(fields)

# ── Load Model & Create UDF ────────────────────────────────────────
def create_fraud_udf(model_path="models/fraud_model.pkl"):
    model = joblib.load(model_path)
    feature_cols = [f"V{i}" for i in range(1, 29)] + \
                   ["Amount_scaled", "Time_scaled"]

    def predict_fraud(*args):
        features = np.array(args).reshape(1, -1)
        prob = model.predict_proba(features)[0][1]
        return float(prob)

    return udf(predict_fraud, DoubleType()), feature_cols

# ── Streaming Pipeline ─────────────────────────────────────────────
def run_streaming(spark, fraud_udf, feature_cols):
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "transactions") \
        .option("startingOffsets", "latest") \
        .load()

    schema = get_schema()

    parsed = raw_stream.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")

    parsed_with_pred = parsed.withColumn(
        "fraud_probability",
        fraud_udf(*[col(c) for c in feature_cols])
    ).withColumn(
        "prediction",
        when(col("fraud_probability") >= 0.5, 1).otherwise(0)
    ).withColumn(
        "alert",
        when(col("fraud_probability") >= 0.5, "FRAUD").otherwise("LEGIT")
    ).withColumn(
        "processed_at", current_timestamp()
    )

    output = parsed_with_pred.select(
        "transaction_id",
        "fraud_probability",
        "prediction",
        "alert",
        "Class",
        "processed_at"
    )

    # Sink 1 — Console
    console_query = output.writeStream \
        .format("console") \
        .option("truncate", False) \
        .option("numRows", 20) \
        .trigger(processingTime="5 seconds") \
        .start()

    # Sink 2 — Delta Lake
    delta_query = output.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", "checkpoints/delta") \
        .trigger(processingTime="5 seconds") \
        .start("delta_output/predictions")

    print("\nStreaming started!")
    print("Console output every 5 seconds")
    print("Predictions saving to delta_output/predictions")
    print("Press Ctrl+C to stop\n")

    spark.streams.awaitAnyTermination()

# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    spark = create_spark_session()
    fraud_udf, feature_cols = create_fraud_udf()
    run_streaming(spark, fraud_udf, feature_cols)
