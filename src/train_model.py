import pandas as pd
import numpy as np
import os
import glob
import joblib

from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb

INPUT_DIR  = 'data/labeled'
MODEL_DIR  = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    'price_change_1h',
    'price_change_6h',
    'price_change_24h',
    'vol_spike_ratio',
    'price_reversal_pct',
    'hl_spread_pct',
    'upper_wick_pct',
    'volatility_6h',
    'volatility_24h',
]

TARGET_COL = 'is_pump'


# ── Load Data ────────────────────────────────────────────────────────────────
def load_all_coins():

    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))

    if not csv_files:
        raise FileNotFoundError(
            f"No labeled CSVs found in '{INPUT_DIR}'. Run label_data.py first."
        )

    dfs = []

    for filepath in csv_files:
        coin = os.path.basename(filepath).replace('_labeled.csv', '')
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        df['coin'] = coin
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values('timestamp').reset_index(drop=True)

    print(f"Loaded {len(data):,} total candles across {data['coin'].nunique()} coins.")
    print(f"Total pump labels: {data[TARGET_COL].sum()} ({data[TARGET_COL].mean()*100:.3f}%)\n")

    return data


# ── Train/Test Split ─────────────────────────────────────────────────────────
def split_data(data):

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


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(name, y_test, y_pred, y_proba):

    print("="*50)
    print(f"  {name}")
    print("="*50)

    print(classification_report(
        y_test,
        y_pred,
        target_names=['Normal', 'Pump'],
        zero_division=0
    ))

    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}\n")


# ── Custom Logistic Regression ───────────────────────────────────────────────
def train_logistic_regression(X_train, y_train, lr=0.1, epochs=1000):

    X = X_train.values
    y = y_train.values

    # feature scaling
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    X = (X - mean) / std

    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0

    # class weighting (balanced)
    n_pos = y.sum()
    n_neg = len(y) - n_pos

    w_pos = len(y) / (2 * n_pos)
    w_neg = len(y) / (2 * n_neg)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for epoch in range(epochs):

        linear = np.dot(X, weights) + bias
        y_pred = sigmoid(linear)

        errors = y_pred - y
        weights_vector = np.where(y == 1, w_pos, w_neg)

        dw = (1/n_samples) * np.dot(X.T, errors * weights_vector)
        db = (1/n_samples) * np.sum(errors * weights_vector)

        weights -= lr * dw
        bias -= lr * db

        if epoch % 200 == 0:
            loss = -np.mean(
                weights_vector * (
                    y*np.log(y_pred + 1e-9) +
                    (1-y)*np.log(1-y_pred + 1e-9)
                )
            )
            print(f"Epoch {epoch} Loss {loss:.4f}")

    return weights, bias, mean, std


def logistic_predict(X, weights, bias, mean, std, threshold=0.8):

    X = X.values
    X = (X - mean) / std

    probs = 1 / (1 + np.exp(-(np.dot(X, weights) + bias)))

    preds = (probs >= threshold).astype(int)

    return preds, probs


# ── Random Forest ────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train):

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_sm, y_train_sm)

    return model


# ── XGBoost ──────────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train):

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


# ── Feature Importance ───────────────────────────────────────────────────────
def print_feature_importance(model):

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False)

    print("XGBoost Feature Importances:")

    for feat, score in importances.items():
        bar = '█' * int(score * 200)
        print(f"  {feat:<22} {score:.4f}  {bar}")

    print()


# ── Main Training Pipeline ───────────────────────────────────────────────────
def main():

    data = load_all_coins()

    X_train, X_test, y_train, y_test = split_data(data)

    # Logistic Regression
    print("Training Logistic Regression...")

    weights, bias, mean, std = train_logistic_regression(X_train, y_train)

    lr_pred, lr_proba = logistic_predict(
        X_test, weights, bias, mean, std, threshold=0.8
    )

    evaluate("Logistic Regression", y_test, lr_pred, lr_proba)

    # Random Forest
    print("Training Random Forest...")

    rf_model = train_random_forest(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:,1]

    evaluate("Random Forest", y_test, rf_pred, rf_proba)

    # XGBoost
    print("Training XGBoost...")

    xgb_model = train_xgboost(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:,1]

    evaluate("XGBoost", y_test, xgb_pred, xgb_proba)

    print_feature_importance(xgb_model)

    # save models
    joblib.dump(rf_model, os.path.join(MODEL_DIR, 'random_forest.pkl'))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgboost.pkl'))

    print("Models saved:")
    print("  models/random_forest.pkl")
    print("  models/xgboost.pkl")


if __name__ == '__main__':
    main()
