import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timezone

COINS = [
    'XRP/USDT', 'DOGE/USDT', 'SHIB/USDT', 'LTC/USDT',
    'ADA/USDT', 'TRX/USDT', 'MATIC/USDT', 'DOT/USDT',
    'LINK/USDT', 'XLM/USDT', 'VET/USDT', 'ALGO/USDT',
    'SAND/USDT', 'MANA/USDT', 'CHZ/USDT', 'HOT/USDT',
    'ANKR/USDT', 'COTI/USDT'
]

OUTPUT_DIR = 'data/raw'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_ohlcv(exchange, symbol, timeframe='1h'):
    since_ms = exchange.parse8601(
        datetime(datetime.now().year - 1,
                 datetime.now().month,
                 datetime.now().day,
                 tzinfo=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    )

    all_candles = []
    while True:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe,
                since=since_ms, limit=1000
            )
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            break

        if not candles:
            break

        all_candles.extend(candles)
        last_ts = candles[-1][0]

        if last_ts >= exchange.milliseconds() - 3600000:
            break

        since_ms = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.drop_duplicates(subset='timestamp', inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def main():
    exchange = ccxt.binanceus({'enableRateLimit': True})

    for symbol in COINS:
        coin_name = symbol.replace('/', '_')
        filepath = os.path.join(OUTPUT_DIR, f'{coin_name}_1h.csv')

        if os.path.exists(filepath):
            print(f"Already exists, skipping: {filepath}")
            continue

        print(f"Fetching {symbol}...")
        df = fetch_ohlcv(exchange, symbol)

        if df.empty:
            print(f"  No data returned for {symbol}")
            continue

        df.to_csv(filepath, index=False)
        print(f"  Saved {len(df)} rows → {filepath}")
        time.sleep(1)

if __name__ == '__main__':
    main()

