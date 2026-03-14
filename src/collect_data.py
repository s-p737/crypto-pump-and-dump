import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timezone

# coins we're pulling from Binance
COINS = [
    # original coins
    'XRP/USDT', 'DOGE/USDT', 'SHIB/USDT', 'LTC/USDT',
    'ADA/USDT', 'TRX/USDT', 'MATIC/USDT', 'DOT/USDT',
    'LINK/USDT', 'XLM/USDT', 'VET/USDT', 'ALGO/USDT',
    'SAND/USDT', 'MANA/USDT', 'CHZ/USDT', 'HOT/USDT',
    'ANKR/USDT', 'COTI/USDT',

    # batch 2
    'FUN/USDT', 'RLC/USDT', 'MTL/USDT', 'GTO/USDT',
    'QSP/USDT', 'BLZ/USDT', 'CND/USDT', 'POA/USDT',
    'STORM/USDT', 'TNT/USDT', 'OST/USDT', 'AST/USDT',
    'SNM/USDT', 'LEND/USDT', 'VIBE/USDT', 'WABI/USDT',
    'GVT/USDT', 'INS/USDT', 'PIVX/USDT', 'STEEM/USDT',
    'NPXS/USDT', 'KEY/USDT', 'NAS/USDT', 'HC/USDT',
    'GO/USDT', 'APPC/USDT', 'DOCK/USDT', 'OAX/USDT',

    # batch 3
    'ZRX/USDT', 'ENJ/USDT', 'ICX/USDT', 'ONT/USDT',
    'ZIL/USDT', 'IOST/USDT', 'CELR/USDT', 'REEF/USDT',
    'SLP/USDT', 'ALICE/USDT', 'TLM/USDT', 'CTSI/USDT',
    'AUDIO/USDT', 'FLOW/USDT', 'SKL/USDT', 'POND/USDT',
    'RVN/USDT', 'DGB/USDT', 'QTUM/USDT', 'ONE/USDT',
    'ROSE/USDT', 'DUSK/USDT', 'GLMR/USDT', 'KAVA/USDT',
    'KSM/USDT', 'BAND/USDT', 'LRC/USDT', 'OGN/USDT',
    'NMR/USDT', 'GTC/USDT', 'REQ/USDT', 'GLM/USDT',

    # batch 4 - everything else confirmed on Binance.US
    'AAVE/USDT', 'UNI/USDT', 'SUSHI/USDT', 'SNX/USDT',
    'CRV/USDT', 'BAT/USDT', 'ZEC/USDT', 'DASH/USDT',
    'ICX/USDT', 'STORJ/USDT', 'SKY/USDT', 'COMP/USDT',
    'MKR/USDT', 'YFI/USDT', 'BNT/USDT', 'BAL/USDT',
    'ATOM/USDT', 'NEAR/USDT', 'FIL/USDT', 'THETA/USDT',
    'TFUEL/USDT', 'HBAR/USDT', 'EGLD/USDT', 'NEO/USDT',
    'XTZ/USDT', 'ETC/USDT', 'BCH/USDT', 'IOTA/USDT',
    'ONT/USDT', 'QTUM/USDT', 'ZEN/USDT', 'LAZIO/USDT',
    'PORTO/USDT', 'SANTOS/USDT', 'ALPINE/USDT', 'APE/USDT',
    'IMX/USDT', 'MASK/USDT', 'GALA/USDT', 'ENS/USDT',
    'LPT/USDT', 'OCEAN/USDT', 'FLUX/USDT', 'RVN/USDT',
    'XNO/USDT', 'WAXP/USDT', 'ORBS/USDT', 'DIA/USDT',
    'BICO/USDT', 'API3/USDT', 'FORTH/USDT', 'ACH/USDT',
    'TWT/USDT', 'CAKE/USDT', 'SUI/USDT', 'APT/USDT',
    'ARB/USDT', 'OP/USDT', 'INJ/USDT', 'SEI/USDT',
    'TIA/USDT', 'JUP/USDT', 'JTO/USDT', 'RENDER/USDT',
    'WIF/USDT', 'BONK/USDT', 'PEPE/USDT', 'FLOKI/USDT',
]
# where the csv's will be saved
OUTPUT_DIR = 'data/raw'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_ohlcv(exchange, symbol, timeframe='1h'):
    """
    Fetches 12 months of hourly OHLCV data for a single coin.
    OHLCV = Open, High, Low, Close, Volume — one row per hour.
    
    Binance only returns 1000 candles per request, so we loop
    and keep fetching the next batch until we reach today.
    """
    
    # calculate the timestamp for exactly 1 year ago 
    since_ms = exchange.parse8601(
        datetime(datetime.now().year - 1,
                 datetime.now().month,
                 datetime.now().day,
                 tzinfo=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    )

    all_candles = []
    
    # keep fetching until we reach the present 
    while True:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe,
                since=since_ms, limit=1000
            )
        except Exception as e:
            # if the API call fails, skip this coin
            print(f"  Error fetching {symbol}: {e}")
            break
        
        # if no candles returned, we've reached the end of available data
        if not candles:
            break

        
        all_candles.extend(candles)
        
        # timestamp of the last candle in this batch (in milliseconds)
        last_ts = candles[-1][0]

        # if the last candle is within 1 hour of now, we've caught up to present
        if last_ts >= exchange.milliseconds() - 3600000:
            break

        # move the start time forward to just after the last candle we fetched
        since_ms = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.drop_duplicates(subset='timestamp', inplace=True)
    df.sort_values('timestamp', inplace=True) # make sure data is sorted oldest → newest
    df.reset_index(drop=True, inplace=True)
    return df

def main():
    # connect to Binance.US (used instead of Binance because Binance blocks US users)
    # enableRateLimit=True tells ccxt to automatically pause between requests
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

