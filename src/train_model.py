import pandas as pd
import numpy as np
import os
import glob
import joblib

from random_forest_scratch import RandomForestScratch
from logistic_regression_scratch import LogisticRegressionScratch
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

INPUT_DIR  = 'data/labeled'
MODEL_DIR  = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    'price_change_1h', 'price_change_6h', 'price_change_24h',
    'vol_spike_ratio', 'price_reversal_pct', 'hl_spread_pct',
    'upper_wick_pct', 'volatility_6h', 'volatility_24h',
]
TARGET_COL = 'is_pump'


def load_all_coins():
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No labeled CSVs found in '{INPUT_DIR}'.")
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
    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL])
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test  = data.iloc[split_idx:]
    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test,  y_test  = test[FEATURE_COLS],  test[TARGET_COL]
    print(f"Train: {len(X_train):,} candles | pumps: {y_train.sum()}")
    print(f"Test:  {len(X_test):,} candles  | pumps: {y_test.sum()}\n")
    return X_train, X_test, y_train, y_test


def evaluate(name, y_true, y_pred, y_proba, split='Test'):
    print(f"{'='*50}")
    print(f"  {name} [{split}]")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pump'], zero_division=0))
    print(f"  ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}\n")


def evaluate_train(name, model, X_train, y_train, use_values=False):
    """
    Evaluates model on training set to check for overfitting.
    If train performance >> test performance, the model has overfit.
    """
    print(f"--- {name} Train Performance ---")
    X = X_train.values if use_values else X_train
    y = y_train.values if use_values else y_train
    pred  = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    print(f"  Train ROC-AUC: {roc_auc_score(y, proba):.4f}")
    print(f"  Train F1:      {f1_score(y, pred, zero_division=0):.4f}")
    print(f"  Train Recall:  {recall_score(y, pred, zero_division=0):.4f}")
    print(f"  Train Prec:    {precision_score(y, pred, zero_division=0):.4f}\n")


def train_logistic_regression(X_train, y_train):
    model = LogisticRegressionScratch(
        learning_rate=0.1, n_iterations=1000,
        random_state=42, class_weight='balanced'
    )
    model.fit(X_train.values, y_train.values)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestScratch(
        n_estimators=100, max_depth=10,
        min_samples_split=5, random_state=42
    )
    model.fit(X_train.values, y_train.values)
    return model


def train_xgboost(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE: {y_train_sm.sum()} pump examples (was {y_train.sum()})")
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, verbosity=0
    )
    model.fit(X_train_sm, y_train_sm)
    return model


def print_feature_importance(model):
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False)
    print("XGBoost Feature Importances:")
    for feat, score in importances.items():
        bar = '█' * int(score * 200)
        print(f"  {feat:<22} {score:.4f}  {bar}")
    print()


def main():
    data = load_all_coins()
    X_train, X_test, y_train, y_test = split_data(data)

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_train("Logistic Regression", lr_model, X_train, y_train, use_values=True)
    lr_pred  = lr_model.predict(X_test.values)
    lr_proba = lr_model.predict_proba(X_test.values)[:, 1]
    evaluate("Logistic Regression", y_test, lr_pred, lr_proba)

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("Training Random Forest (from scratch)...")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_train("Random Forest", rf_model, X_train, y_train, use_values=True)
    rf_pred  = rf_model.predict(X_test.values)
    rf_proba = rf_model.predict_proba(X_test.values)[:, 1]
    evaluate("Random Forest", y_test, rf_pred, rf_proba)
    joblib.dump(rf_model, os.path.join(MODEL_DIR, 'random_forest.pkl'))

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_train("XGBoost", xgb_model, X_train, y_train, use_values=False)
    xgb_pred  = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    evaluate("XGBoost", y_test, xgb_pred, xgb_proba)

    # ── Feature importances ───────────────────────────────────────────────────
    print_feature_importance(xgb_model)

    # ── Save models ───────────────────────────────────────────────────────────
    joblib.dump(lr_model,  os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgboost.pkl'))
    print("Models saved to models/")


if __name__ == '__main__':
    main()