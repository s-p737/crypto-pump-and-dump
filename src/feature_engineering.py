# feature_engineering.py
# CS229 Final Project — Detecting Pump-and-Dump Schemes in Cryptocurrency Markets
# Stuti Patel (snpatel7) & Meghan D'Souza (megands)
#
# Computes pump-and-dump signal features from raw OHLCV candle data.
#
# Input:  data/raw/*.csv         (1h candles fetched from exchange APIs)
# Output: data/features/*.csv    (feature-engineered candles for label_data.py)

import pandas as pd
import numpy as np
import os
import glob

INPUT_DIR  = 'data/raw'
OUTPUT_DIR = 'data/features'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# computes pump-and-dump signal features from raw OHLCV data
# all rolling calculations are done on a per-coin basis
def engineer_features(df):
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # ── Log returns (used for volatility) ────────────────────────────────────
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # ── Price change % over rolling windows ──────────────────────────────────
    for w in [1, 6, 24]:
        df[f'price_change_{w}h'] = df['close'].pct_change(w) * 100

    # ── Volume spike ratio: current volume vs. rolling 24-candle mean ────────
    rolling_vol_mean = df['volume'].rolling(24).mean()
    df['vol_spike_ratio'] = df['volume'] / (rolling_vol_mean + 1e-9)

    # ── Price reversal: how far close fell from the candle's high ─────────────
    # high value = price pumped up but dumped before candle closed
    df['price_reversal_pct'] = ((df['high'] - df['close']) / (df['high'] + 1e-9)) * 100

    # ── Volatility: rolling std of log returns ────────────────────────────────
    for w in [6, 24]:
        df[f'volatility_{w}h'] = df['log_return'].rolling(w).std() * 100

    # ── High-low spread: wide candles signal volatility or wash trading ───────
    df['hl_spread_pct'] = ((df['high'] - df['low']) / (df['close'] + 1e-9)) * 100

    # ── Upper wick: long upper wick = selling pressure after a pump ───────────
    candle_top = df[['open', 'close']].max(axis=1)
    df['upper_wick_pct'] = ((df['high'] - candle_top) / (df['close'] + 1e-9)) * 100

    return df


def main():
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))

    if not csv_files:
        print(f"No CSV files found in '{INPUT_DIR}'. Run your data fetching script first.")
        return

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")

        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        df = engineer_features(df)

        out_path = os.path.join(OUTPUT_DIR, filename.replace('_1h.csv', '_features.csv'))
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows → {out_path}")


if __name__ == '__main__':
    main()