import pandas as pd
import numpy as np
import os
import glob
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)

INPUT_DIR = 'data/labeled'
MODEL_DIR  = 'models'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    'price_change_1h', 'price_change_6h', 'price_change_24h',
    'vol_spike_ratio', 'price_reversal_pct', 'hl_spread_pct',
    'upper_wick_pct', 'volatility_6h', 'volatility_24h',
]
TARGET_COL = 'is_pump'

# ── Liquidity tiers ───────────────────────────────────────────────────────────
# Large-cap coins are harder to manipulate due to high liquidity.
# We compare model performance across tiers to test this hypothesis.
LARGE_CAP  = {'XRP', 'ADA', 'DOGE', 'LTC', 'DOT', 'LINK', 'XLM', 'ALGO', 'SHIB'}
MEDIUM_CAP = {'SAND', 'MANA', 'CHZ', 'VET', 'ANKR', 'COTI', 'SUSHI', 'UNI',
              'AAVE', 'SNX', 'CRV', 'ATOM', 'NEAR', 'FIL', 'THETA'}


def get_tier(coin):
    """Assigns a liquidity tier label to a coin."""
    if coin in LARGE_CAP:
        return 'large_cap'
    elif coin in MEDIUM_CAP:
        return 'medium_cap'
    else:
        return 'small_cap'


def load_data():
    """Loads all labeled CSVs, adds coin and tier columns, sorts by time."""
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    dfs = []
    for filepath in csv_files:
        coin = os.path.basename(filepath).replace('_labeled.csv', '')
        df   = pd.read_csv(filepath, parse_dates=['timestamp'])
        df['coin'] = coin
        df['tier'] = get_tier(coin)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values('timestamp').reset_index(drop=True)
    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return data


def get_test_set(data):
    """Returns the same 20% chronological test split used in training."""
    split_idx = int(len(data) * 0.8)
    return data.iloc[split_idx:]


# ── Plot 1: ROC Curve comparison ──────────────────────────────────────────────
def plot_roc_curves(y_test, lr_proba, xgb_proba):
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, lr_proba,  name='Logistic Regression', ax=ax)
    RocCurveDisplay.from_predictions(y_test, xgb_proba, name='XGBoost + SMOTE',      ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random baseline')
    ax.set_title('ROC Curve: Logistic Regression vs XGBoost')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")


# ── Plot 2: Confusion matrix ───────────────────────────────────────────────────
def plot_confusion_matrix(y_test, xgb_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, xgb_pred,
        display_labels=['Normal', 'Pump'],
        ax=ax, colorbar=False, cmap='Blues'
    )
    ax.set_title('XGBoost Confusion Matrix (Test Set)')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")


# ── Plot 3: PCA visualization ──────────────────────────────────────────────────
def plot_pca(data):
    """
    Reduces 9 features to 2 principal components and plots pump vs normal.
    If pump events form a distinct cluster, it validates that our features
    capture meaningful signal and that the classes are geometrically separable.
    """
    X = data[FEATURE_COLS]
    y = data[TARGET_COL]

    # scale first — PCA is sensitive to feature magnitude
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['is_pump'] = y.values

    fig, ax = plt.subplots(figsize=(10, 7))

    # plot normal candles (subsample so plot isn't overwhelmed)
    normal = pca_df[pca_df['is_pump'] == 0].sample(5000, random_state=42)
    pumps  = pca_df[pca_df['is_pump'] == 1]

    ax.scatter(normal['PC1'], normal['PC2'],
               c='steelblue', alpha=0.2, s=5, label='Normal', rasterized=True)
    ax.scatter(pumps['PC1'],  pumps['PC2'],
               c='crimson',   alpha=0.8, s=20, label='Pump', zorder=5)

    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f'PC1 ({var[0]:.1f}% variance explained)')
    ax.set_ylabel(f'PC2 ({var[1]:.1f}% variance explained)')
    ax.set_title('PCA: Pump vs Normal Candles')
    ax.legend(markerscale=3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'pca_visualization.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")

    # print how much variance the first 2 components explain
    print(f"\nPCA variance explained: PC1={var[0]:.1f}%, PC2={var[1]:.1f}%, "
          f"Total={sum(var):.1f}%")


# ── Plot 4: Performance by liquidity tier ─────────────────────────────────────
def plot_tier_performance(test_data, xgb_model):
    """
    Breaks down XGBoost performance by coin liquidity tier.
    Tests the hypothesis that pump detection is easier on smaller coins.
    """
    results = []

    for tier in ['large_cap', 'medium_cap', 'small_cap']:
        tier_data = test_data[test_data['tier'] == tier]
        if tier_data[TARGET_COL].sum() == 0:
            continue

        X_tier = tier_data[FEATURE_COLS]
        y_tier = tier_data[TARGET_COL]
        pred   = xgb_model.predict(X_tier)
        proba  = xgb_model.predict_proba(X_tier)[:, 1]

        results.append({
            'Tier':      tier,
            'Coins':     tier_data['coin'].nunique(),
            'Pumps':     y_tier.sum(),
            'Precision': precision_score(y_tier, pred, zero_division=0),
            'Recall':    recall_score(y_tier, pred,    zero_division=0),
            'F1':        f1_score(y_tier, pred,        zero_division=0),
            'ROC-AUC':   roc_auc_score(y_tier, proba)
        })

    results_df = pd.DataFrame(results)
    print("\nPerformance by Liquidity Tier:")
    print(results_df.to_string(index=False))

    # bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x     = np.arange(len(results_df))
    width = 0.25
    ax.bar(x - width, results_df['Precision'], width, label='Precision', color='steelblue')
    ax.bar(x,         results_df['Recall'],    width, label='Recall',    color='coral')
    ax.bar(x + width, results_df['F1'],        width, label='F1',        color='mediumseagreen')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Tier'])
    ax.set_ylim(0, 1.1)
    ax.set_title('XGBoost Performance by Coin Liquidity Tier')
    ax.set_ylabel('Score')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'tier_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")


def main():
    # ── Load data and models ──────────────────────────────────────────────────
    print("Loading data...")
    data = load_data()
    test_data = get_test_set(data)

    X_test = test_data[FEATURE_COLS]
    y_test = test_data[TARGET_COL]

    print(f"Test set: {len(test_data):,} candles | {y_test.sum()} pumps\n")

    lr_model  = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgboost.pkl'))

    # ── Get predictions ───────────────────────────────────────────────────────
    lr_pred   = lr_model.predict(X_test)
    lr_proba  = lr_model.predict_proba(X_test)[:, 1]
    xgb_pred  = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    # ── Print final evaluation ────────────────────────────────────────────────
    print("=== Final Evaluation ===\n")
    for name, pred, proba in [
        ('Logistic Regression', lr_pred, lr_proba),
        ('XGBoost + SMOTE',     xgb_pred, xgb_proba)
    ]:
        print(f"--- {name} ---")
        print(classification_report(y_test, pred,
              target_names=['Normal', 'Pump'], zero_division=0))
        print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}\n")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("Generating plots...")
    plot_roc_curves(y_test, lr_proba, xgb_proba)
    plot_confusion_matrix(y_test, xgb_pred)
    plot_pca(data)
    plot_tier_performance(test_data, xgb_model)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()