"""
Insurance Premium Prediction System - model training.

Run:  python train_model.py
- Uses insurance.csv if present (Kaggle "Medical Cost Personal Datasets"),
  otherwise generates a synthetic dataset with the same columns.
- Trains Linear Regression, Random Forest and Gradient Boosting,
  prints metrics, and saves the best model to model.pkl.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "insurance.csv"
MODEL_PATH = "model.pkl"

NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]


def generate_synthetic_data(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 65, n),
        "sex": rng.choice(["male", "female"], n),
        "bmi": np.round(rng.normal(30, 6, n).clip(15, 50), 1),
        "children": rng.integers(0, 6, n),
        "smoker": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
        "region": rng.choice(["northeast", "northwest", "southeast", "southwest"], n),
    })
    charges = (
        2000
        + df["age"] * 260
        + (df["bmi"] - 25).clip(lower=0) * 300
        + df["children"] * 450
        + (df["smoker"] == "yes") * 23000
        + (df["smoker"].eq("yes") & df["bmi"].gt(30)) * 12000
        + rng.normal(0, 2500, n)
    )
    df["charges"] = charges.clip(lower=1200).round(2)
    return df


def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Loaded {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    print("insurance.csv not found -> generating synthetic data")
    df = generate_synthetic_data()
    df.to_csv(DATA_PATH, index=False)
    return df


def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df = load_data().dropna()
    X, y = df[NUMERIC + CATEGORICAL], df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    best_name, best_pipe, best_r2 = None, None, -np.inf
    for name, model in candidates.items():
        pipe = build_pipeline(model).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        r2, mae = r2_score(y_test, pred), mean_absolute_error(y_test, pred)
        print(f"{name:20s} R2 = {r2:.3f}   MAE = {mae:,.0f}")
        if r2 > best_r2:
            best_name, best_pipe, best_r2 = name, pipe, r2

    joblib.dump(best_pipe, MODEL_PATH)
    print(f"\nBest model: {best_name} (R2 = {best_r2:.3f}) -> saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
