import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

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


def plot_confusion_matrix(y_test, preds):

    cm = confusion_matrix(y_test, preds)

    tn, fp, fn, tp = cm.ravel()

    print("\n=== CONFUSION MATRIX ===")
    print(cm)
    print(f"\nTN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print(f"Total: {cm.sum()}")

    fig, ax = plt.subplots(figsize=(6,5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Pump"]
    )

    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("XGBoost Confusion Matrix")

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

    plt.savefig(path, dpi=300)

    plt.close()

    print(f"Saved → {path}")


def plot_roc(y_test, proba):

    fig, ax = plt.subplots(figsize=(8,6))

    RocCurveDisplay.from_predictions(
        y_test,
        proba,
        name="XGBoost",
        ax=ax
    )

    ax.set_title("ROC Curve")

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "roc_curve.png")

    plt.savefig(path, dpi=300)

    plt.close()

    print(f"Saved → {path}")


def plot_feature_importance(model):

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8,6))

    importance.plot(kind="barh", ax=ax)

    ax.set_title("XGBoost Feature Importance")

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "feature_importance.png")

    plt.savefig(path, dpi=300)

    plt.close()

    print(f"Saved → {path}")


def main():

    print("\nLoading labeled data...")

    data = load_data()

    print(f"Total candles: {len(data):,}")
    print(f"Total pumps:   {data[TARGET_COL].sum():,}")

    split_idx = int(len(data) * 0.8)

    train_data = data.iloc[:split_idx]
    test_data  = data.iloc[split_idx:]

    print("\n=== DATA SPLIT ===")
    print(f"Train candles: {len(train_data):,}")
    print(f"Train pumps:   {train_data[TARGET_COL].sum():,}")

    print(f"\nTest candles:  {len(test_data):,}")
    print(f"Test pumps:    {test_data[TARGET_COL].sum():,}")

    X_test = test_data[FEATURE_COLS]
    y_test = test_data[TARGET_COL]

    print("\nLoading model...")

    model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]

    print("\n=== PREDICTION CHECK ===")
    print(f"Test candles:      {len(y_test)}")
    print(f"Actual pumps:      {y_test.sum()}")
    print(f"Predicted pumps:   {preds.sum()}")

    plot_confusion_matrix(y_test, preds)

    plot_roc(y_test, proba)

    plot_feature_importance(model)


if __name__ == "__main__":
    main()