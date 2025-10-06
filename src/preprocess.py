# src/preprocess.py
import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/raw/housing.csv"
PROCESSED_DATA_DIR = "data/processed"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "housing_processed.csv")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Load raw data
if os.path.exists(RAW_DATA_PATH):
    df = pd.read_csv(RAW_DATA_PATH)
else:
    # Generate synthetic data if raw data is missing
    print(f"[INFO] {RAW_DATA_PATH} not found. Generating synthetic data...")
    X = np.random.randn(200, 8)
    y = X.dot(np.random.randn(8)) + np.random.randn(200) * 0.1
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
    df['target'] = y
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"[INFO] Synthetic raw data saved at {RAW_DATA_PATH}")

# Optionally, do preprocessing here (e.g., normalize, fill missing values)
# For now, just save as processed CSV
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"[INFO] Processed data saved at {PROCESSED_DATA_PATH}")
