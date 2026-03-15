import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)

# -----------------------------
# Paths
# -----------------------------

INPUT_DIR = "../data/labeled"
MODEL_DIR = "../models"
OUTPUT_DIR = "../outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "is_pump"

FEATURE_COLS = [
    "price_change_1h",
    "price_change_6h",
    "price_change_24h",
    "vol_spike_ratio",
    "price_reversal_pct",
    "hl_spread_pct",
    "upper_wick_pct",
    "volatility_6h",
    "volatility_24h"
]

# -----------------------------
# Load labeled data
# -----------------------------

def load_data():

    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

    dfs = []

    for f in files:

        coin = os.path.basename(f).replace("_labeled.csv", "")

        df = pd.read_csv(f, parse_dates=["timestamp"])
        df["coin"] = coin

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    data = data.sort_values("timestamp")

    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL])

    return data


# -----------------------------
# Plot confusion matrix
# -----------------------------

def plot_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6,5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal","Pump"]
    )

    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("XGBoost Confusion Matrix")

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

    plt.savefig(path, dpi=300)

    plt.close()

    print(f"Saved → {path}")

    return cm


# -----------------------------
# Plot ROC curve
# -----------------------------

def plot_roc(y_test, xgb_proba):

    fig, ax = plt.subplots(figsize=(8,6))

    RocCurveDisplay.from_predictions(
        y_test,
        xgb_proba,
        name="XGBoost",
        ax=ax
    )

    ax.set_title("ROC Curve")

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR,"roc_curve.png")

    plt.savefig(path, dpi=300)

    plt.close()

    print(f"Saved → {path}")


# -----------------------------
# Main evaluation
# -----------------------------

def main():

    print("\nLoading labeled dataset...")

    data = load_data()

    print(f"Total candles: {len(data):,}")
    print(f"Total pumps:   {data[TARGET_COL].sum():,}\n")

    # -----------------------------
    # Train/Test Split (chronological)
    # -----------------------------

    split_idx = int(len(data) * 0.8)

    train_data = data.iloc[:split_idx]
    test_data  = data.iloc[split_idx:]

    print("=== DATASET SPLIT ===")
    print(f"Train candles: {len(train_data):,}")
    print(f"Train pumps:   {train_data[TARGET_COL].sum():,}")

    print(f"\nTest candles:  {len(test_data):,}")
    print(f"Test pumps:    {test_data[TARGET_COL].sum():,}\n")

    # -----------------------------
    # Prepare test data
    # -----------------------------

    X_test = test_data[FEATURE_COLS]
    y_test = test_data[TARGET_COL]

    # -----------------------------
    # Load trained model
    # -----------------------------

    model_path = os.path.join(MODEL_DIR, "xgboost.pkl")

    print("Loading model...")

    xgb_model = joblib.load(model_path)

    # -----------------------------
    # Predictions
    # -----------------------------

    xgb_pred  = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:,1]

    print("=== PREDICTION CHECK ===")

    print(f"Test candles:      {len(y_test):,}")
    print(f"Actual pumps:      {y_test.sum():,}")
    print(f"Predicted pumps:   {xgb_pred.sum():,}\n")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------

    cm = plot_confusion_matrix(y_test, xgb_pred)

    tn, fp, fn, tp = cm.ravel()

    print("=== CONFUSION MATRIX ===")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    print(f"\nTotal samples: {cm.sum():,}")

    # sanity check
    if cm.sum() != len(y_test):
        print("WARNING: confusion matrix does not match test size")

    # -----------------------------
    # ROC Curve
    # -----------------------------

    plot_roc(y_test, xgb_proba)

    # -----------------------------
    # Classification report
    # -----------------------------

    print("\n=== CLASSIFICATION REPORT ===")

    print(classification_report(y_test, xgb_pred))


# -----------------------------

if __name__ == "__main__":
    main()