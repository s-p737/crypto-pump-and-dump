import pandas as pd
import numpy as np
import os
import glob
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb

INPUT_DIR  = 'data/labeled'
MODEL_DIR  = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Features the model will train on ─────────────────────────────────────────
# These are all backward-looking — they only use information available
# at the time of the candle, so there is no data leakage.
#
# We explicitly EXCLUDE 'future_return_6h' which was used during labeling
# because that looks into the future — the model would never have access
# to that in a real-world prediction scenario.
FEATURE_COLS = [
    'price_change_1h',    # how much price moved in last 1 hour
    'price_change_6h',    # how much price moved in last 6 hours
    'price_change_24h',   # how much price moved in last 24 hours
    'vol_spike_ratio',    # current volume vs 24h rolling mean
    'price_reversal_pct', # how far close fell from candle's high
    'hl_spread_pct',      # high-low range as % of close (intra-candle volatility)
    'upper_wick_pct',     # upper wick size — selling pressure after a spike
    'volatility_6h',      # rolling std of log returns over 6 hours
    'volatility_24h',     # rolling std of log returns over 24 hours
]

TARGET_COL = 'is_pump'


def load_all_coins():
    """
    Loads and concatenates labeled CSVs from all coins into one dataframe.
    Sorting by timestamp ensures our train/test split respects time order.
    """
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))

    if not csv_files:
        raise FileNotFoundError(f"No labeled CSVs found in '{INPUT_DIR}'. Run label_data.py first.")

    dfs = []
    for filepath in csv_files:
        coin = os.path.basename(filepath).replace('_labeled.csv', '')
        df   = pd.read_csv(filepath, parse_dates=['timestamp'])
        df['coin'] = coin
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values('timestamp').reset_index(drop=True)

    print(f"Loaded {len(data):,} total candles across {data['coin'].nunique()} coins.")
    print(f"Total pump labels: {data[TARGET_COL].sum()} ({data[TARGET_COL].mean()*100:.3f}%)\n")

    return data


def split_data(data):
    """
    Splits data into train (80%) and test (20%) sets chronologically.

    We split by TIME, not randomly, because:
    - In reality you train on historical data and predict on future data
    - A random split would let the model see future candles during training
      (data leakage), which would make results look better than they are
    """
    # drop rows missing any feature or the target
    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL])

    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test  = data.iloc[split_idx:]

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET_COL]

    print(f"Train: {len(X_train):,} candles | pumps: {y_train.sum()}")
    print(f"Test:  {len(X_test):,} candles  | pumps: {y_test.sum()}\n")

    return X_train, X_test, y_train, y_test


def evaluate(name, y_test, y_pred, y_proba):
    """Prints a full evaluation report for a model."""
    print(f"{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Pump'], zero_division=0))
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}\n")


def train_logistic_regression(X_train, y_train):
    """
    Logistic Regression baseline.

    We wrap it in a Pipeline with StandardScaler because Logistic Regression
    is sensitive to feature scale — without scaling, features like
    'vol_spike_ratio' (values 0–10) would dominate over 'volatility_6h'
    (values 0.001–0.01) just because of their magnitude.

    class_weight='balanced' tells sklearn to automatically upweight the
    minority class (pumps) to compensate for the heavy class imbalance.
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train):
    """
    XGBoost model with SMOTE oversampling.

    Instead of just telling XGBoost to upweight pump examples via
    scale_pos_weight, SMOTE actually generates synthetic pump examples
    by interpolating between real ones. This gives the model more
    diverse pump patterns to learn from, which should improve precision
    without sacrificing recall.

    Inspired by the credit card fraud detection literature (ULB dataset),
    which faces the same class imbalance problem as ours.
    """
    # apply SMOTE to training data only — never to test data
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    print(f"After SMOTE: {y_train_sm.sum()} pump examples (was {y_train.sum()})")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    model.fit(X_train_sm, y_train_sm)
    return model
def print_feature_importance(model):
    """Prints XGBoost feature importances ranked highest to lowest."""
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False)
    print("XGBoost Feature Importances:")
    for feat, score in importances.items():
        bar = '█' * int(score * 200)
        print(f"  {feat:<22} {score:.4f}  {bar}")
    print()


def main():
    # ── 1. Load all labeled coin data ────────────────────────────────────────
    data = load_all_coins()

    # ── 2. Train/test split (chronological) ──────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(data)

    # ── 3. Train Logistic Regression (baseline) ───────────────────────────────
    print("Training Logistic Regression...")
    lr_model  = train_logistic_regression(X_train, y_train)
    lr_pred   = lr_model.predict(X_test)
    lr_proba  = lr_model.predict_proba(X_test)[:, 1]
    evaluate("Logistic Regression (Baseline)", y_test, lr_pred, lr_proba)

    # ── 4. Train XGBoost ──────────────────────────────────────────────────────
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_pred  = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    evaluate("XGBoost", y_test, xgb_pred, xgb_proba)

    # ── 5. Feature importances ────────────────────────────────────────────────
    print_feature_importance(xgb_model)

    # ── 6. Save models ────────────────────────────────────────────────────────
    # We save both models so they can be loaded later for evaluation/prediction
    # without retraining from scratch.
    joblib.dump(lr_model,  os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgboost.pkl'))
    print("Models saved to models/")
    print("  models/logistic_regression.pkl")
    print("  models/xgboost.pkl")


if __name__ == '__main__':
    main()