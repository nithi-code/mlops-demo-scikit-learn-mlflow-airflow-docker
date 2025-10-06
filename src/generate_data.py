
import numpy as np
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)
rng = np.random.RandomState(42)
X = rng.randn(200, 8)
coef = rng.randn(8)
y = X.dot(coef) + rng.randn(200) * 0.1
cols = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=cols)
df['target'] = y
df.to_csv("data/raw/housing.csv", index=False)
print("Generated synthetic data at data/raw/housing.csv")
