import pandas as pd
import numpy as np
import os
import glob

INPUT_DIR  = 'data/features'
OUTPUT_DIR = 'data/labeled'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Labeling thresholds ───────────────────────────────────────────────────────
# These are based on heuristics from prior research (e.g. Ramos & Golub 2017)
# and common pump-and-dump characteristics observed in crypto markets.
# A candle is labeled as a pump (is_pump = 1) if ALL THREE conditions are met:
#
#   1. Price rose sharply in the preceding 6 hours      → PUMP phase
#   2. Price then dropped sharply in the following 6 hours → DUMP phase
#   3. Volume was abnormally high relative to recent average → manipulation signal
#
# Unlike Ramos & Golub (2017) who had access to order cancellation data to label
# ground truth, we rely on these outcome-based heuristics as a proxy for
# manipulation, since cancellation data is not available through public APIs.

PRICE_UP_THRESHOLD   = 0.10   # price must have risen  >= 20% over last 6 candles
PRICE_DOWN_THRESHOLD = 0.05   # price must then fall   >= 10% over next 6 candles
VOLUME_SPIKE_RATIO   = 2.0    # volume must be         >= 3x the 24h rolling mean


# adds an 'is_pump' column to a feature-engineered dataframe
# uses price_change_6h (backward) + future_return_6h (forward) + vol_spike_ratio
# note: future_return_6h is ONLY used for labeling — it's forward-looking and
# must be dropped before training since we wouldn't know future prices in production
def label_pumps(df):
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # ── Forward-looking return: % price change over the NEXT 6 candles ───────
    # shift(-6) looks 6 rows ahead — this tells us if a dump followed the pump.
    # A negative value means price fell after this candle.
    df['future_return_6h'] = df['close'].pct_change(6).shift(-6)

    # ── Apply all three labeling conditions ───────────────────────────────────
    pump_condition = (
        # Condition 1: price rose >= 20% over the past 6 hours (pump phase)
        (df['price_change_6h']   >= PRICE_UP_THRESHOLD * 100)   &

        # Condition 2: price then fell >= 10% in the next 6 hours (dump phase)
        (df['future_return_6h']  <= -PRICE_DOWN_THRESHOLD)       &

        # Condition 3: volume was >= 3x the rolling 24h average (unusual activity)
        (df['vol_spike_ratio']   >= VOLUME_SPIKE_RATIO)
    )

    # Label: 1 = pump-and-dump, 0 = normal market activity
    df['is_pump'] = pump_condition.astype(int)

    return df


# prints a quick summary of pump labels for a given coin
def summarize(df, coin):
    total  = len(df)
    pumps  = df['is_pump'].sum()
    rate   = pumps / total * 100
    print(f"  {coin}: {pumps} pumps out of {total} candles ({rate:.2f}%)")


def main():
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))

    if not csv_files:
        print(f"No CSV files found in '{INPUT_DIR}'. Run feature_engineering.py first.")
        return

    all_pump_counts = []

    for filepath in csv_files:
        filename  = os.path.basename(filepath)
        coin_name = filename.replace('_features.csv', '')

        print(f"Labeling {coin_name}...")

        df = pd.read_csv(filepath, parse_dates=['timestamp'])

        # need price_change_6h and vol_spike_ratio from feature engineering
        required = ['price_change_6h', 'vol_spike_ratio', 'close']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            print(f"  Skipping — missing columns: {missing}")
            continue

        df = label_pumps(df)

        summarize(df, coin_name)
        all_pump_counts.append(df['is_pump'].sum())

        # save labeled data — future_return_6h is kept here for reference
        # but MUST be dropped in train_model.py before fitting the model
        out_path = os.path.join(OUTPUT_DIR, filename.replace('_features.csv', '_labeled.csv'))
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

    # ── Overall dataset summary ───────────────────────────────────────────────
    print(f"\nTotal pump events labeled across all coins: {sum(all_pump_counts)}")
    print("If pump count is 0, consider lowering PRICE_UP_THRESHOLD or VOLUME_SPIKE_RATIO.")
    print("If pump count is very high (>5%), consider raising thresholds.")


if __name__ == '__main__':
    main()