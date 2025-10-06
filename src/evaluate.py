
import joblib, json, os, pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

model = joblib.load("artifacts/model.joblib")
data = pd.read_csv("data/raw/housing.csv")
X = data.drop(columns=["target"])
y = data["target"]
preds = model.predict(X)
mse = mean_squared_error(y, preds)
r2 = r2_score(y, preds)
print("Evaluation on full data - MSE:", mse, "R2:", r2)
with open("artifacts/eval.json","w") as f:
    json.dump({"mse": mse, "r2": r2}, f)
