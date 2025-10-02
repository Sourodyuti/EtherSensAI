"""
AI Crypto Analyst & Signal Generator (Automated)
Implements:
 - Technical checks (4H, 1H, 15M): EMA50/200, RSI, Stochastic
 - Sentiment check: Fear & Greed Index (alternative.me)
 - Macro check: SPX/DXY optional via yfinance (user-configurable)
 - Signal rules: Technical mandatory + at least one of Sentiment or Macro
 - Outputs structured signal exactly as requested

Usage:
    python ai_crypto_analyst.py
"""

import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import time
from datetime import datetime, timezone
import yfinance as yf
from dateutil import tz

# -------------------------
# User configuration
# -------------------------
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
EXCHANGE_ID = "binance"
# timeframes mapping between human and ccxt
TF_4H = "4h"
TF_1H = "1h"
TF_15M = "15m"

# indicator params
RSI_PERIOD = 14
STOCH_K = 14
STOCH_D = 3
EMA_FAST = 50
EMA_SLOW = 200

# Risk profile defaults
MIN_RR = 3.0  # minimum reward:risk
MAX_SL_DISTANCE_PCT = 0.06  # don't accept setups with SL > 6% by default (for high leverage)
MIN_CONFIDENCE = 0.5

# Macro tickers (optional). If you don't want macro checks, set to None.
DXY_TICKER = None   # e.g. "DX-Y.NYB" (user can change). Set to None to disable.
SPX_TICKER = "^GSPC"

# Optional: NewsAPI key (if you want headline scanning). Leave None to skip.
NEWSAPI_KEY = "b84f417086344a90b1f2ec5237b509f7"

# -------------------------
# Helpers & Data fetchers
# -------------------------
exchange = getattr(ccxt, EXCHANGE_ID)({
    'enableRateLimit': True,
})

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500):
    """
    Returns a pandas DataFrame with columns: timestamp, open, high, low, close, volume
    """
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open','high','low','close','volume']].astype(float)
    return df

def fetch_current_price(symbol: str):
    ticker = exchange.fetch_ticker(symbol)
    return float(ticker['last'])

# -------------------------
# Indicator computations
# -------------------------
def add_indicators(df: pd.DataFrame):
    """ Adds EMA50, EMA200, RSI14, Stochastic %K and %D to df (in-place copy returned) """
    df = df.copy()
    df['ema50'] = ta.ema(df['close'], length=EMA_FAST)
    df['ema200'] = ta.ema(df['close'], length=EMA_SLOW)
    df['rsi'] = ta.rsi(df['close'], length=RSI_PERIOD)
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=STOCH_K, d=STOCH_D)
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']
    return df

# -------------------------
# Technical rule checks
# -------------------------
def trend_via_emas(df_4h, df_1h):
    """
    Simple trend detection:
      - bullish if ema50 > ema200 on both 4H and 1H
      - bearish if ema50 < ema200 on both
      - else neutral
    """
    last_4h = df_4h.iloc[-1]
    last_1h = df_1h.iloc[-1]
    if (last_4h['ema50'] > last_4h['ema200']) and (last_1h['ema50'] > last_1h['ema200']):
        return "bullish"
    if (last_4h['ema50'] < last_4h['ema200']) and (last_1h['ema50'] < last_1h['ema200']):
        return "bearish"
    return "neutral"

def recent_swing_low_high(df, lookback=30):
    highs = df['high'].tail(lookback)
    lows = df['low'].tail(lookback)
    return float(highs.max()), float(lows.min())

def technical_signal(symbol):
    """
    Returns:
      - None if technicals don't provide a clear direction/invalidation
      - dict with direction, entry, sl, suggested_tp based on 1:3 R:R and recent swing levels
    """
    df4 = add_indicators(fetch_ohlcv(symbol, TF_4H, limit=200))
    df1 = add_indicators(fetch_ohlcv(symbol, TF_1H, limit=200))
    df15 = add_indicators(fetch_ohlcv(symbol, TF_15M, limit=200))

    trend = trend_via_emas(df4, df1)
    price = float(df15['close'][-1])

    # define swing levels from 1H for SL/TP anchors
    recent_high, recent_low = recent_swing_low_high(df1, lookback=40)

    # liquidity concept: large cluster above recent high or below recent low
    # For simplicity: treat liquidity zones as the recent high/low +/- small buffer

    # Price-action confirmations on 15m: RSI divergence/stochastic extremes (simple rules)
    rsi_15 = float(df15['rsi'][-1])
    stoch_k_15 = float(df15['stoch_k'][-1])

    # Determine LONG or SHORT candidates:
    if trend == "bullish":
        # favor LONG setups: invalidation = swing low on 1H
        sl = recent_low * 0.995  # slightly below recent low
        sl_dist = (price - sl) / sl
        # compute absolute pct sl
        sl_pct = (price - sl) / price
        if sl_pct <= 0 or sl_pct > MAX_SL_DISTANCE_PCT:
            return None

        # require bullish confirmations: RSI not overbought and Stoch turning up
        cond = (rsi_15 > 30 and rsi_15 < 70) and (stoch_k_15 < 50)
        if not cond:
            return None

        risk = price - sl
        tp = price + (risk * MIN_RR)
        return {
            "direction": "LONG",
            "entry": round(price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "trend": trend,
            "technical_conf": True
        }

    if trend == "bearish":
        # favor SHORT setups: invalidation = swing high on 1H
        sl = recent_high * 1.005  # slightly above recent high
        sl_pct = (sl - price) / price
        if sl_pct <= 0 or sl_pct > MAX_SL_DISTANCE_PCT:
            return None

        cond = (rsi_15 < 70 and stoch_k_15 > 50)
        if not cond:
            return None

        risk = sl - price
        tp = price - (risk * MIN_RR)
        return {
            "direction": "SHORT",
            "entry": round(price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "trend": trend,
            "technical_conf": True
        }

    return None

# -------------------------
# Sentiment & News checks
# -------------------------
def fetch_fear_and_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1")
        if r.status_code == 200:
            j = r.json()
            if 'data' in j and len(j['data']) > 0:
                val = int(j['data'][0]['value'])
                classification = j['data'][0]['value_classification']
                return {"value": val, "classification": classification}
    except Exception as e:
        print("FNG fetch error:", e)
    return None

def sentiment_check():
    fng = fetch_fear_and_greed()
    # simple rule: extreme fear (<25) bullish contrarian; extreme greed (>75) bearish contrarian
    if not fng:
        return None
    val = fng['value']
    if val <= 25:
        return {"signal": "bullish", "value": val, "reason": "extreme fear"}
    elif val >= 75:
        return {"signal": "bearish", "value": val, "reason": "extreme greed"}
    else:
        return {"signal": "neutral", "value": val, "reason": "neutral"}

# -------------------------
# Macro checks (optional)
# -------------------------
def macro_check():
    """
    Returns simple macro reading:
      - bullish if SPX is up on 4H and DXY is falling
      - bearish if SPX down and DXY rising
    This is a simple approximation.
    """
    try:
        spx = None
        dxy = None
        if SPX_TICKER:
            spx_hist = yf.Ticker(SPX_TICKER).history(period="5d", interval="1h")
            spx = float(spx_hist['Close'][-1])
            spx_prev = float(spx_hist['Close'][-5])  # compare to earlier
        if DXY_TICKER:
            dxy_hist = yf.Ticker(DXY_TICKER).history(period="5d", interval="1h")
            dxy = float(dxy_hist['Close'][-1])
            dxy_prev = float(dxy_hist['Close'][-5])

        result = {"spx": spx, "dxy": dxy}
        # heuristic
        if spx and DXY_TICKER and dxy:
            if (spx > spx_prev) and (dxy < dxy_prev):
                return {"signal": "bullish", "reason": "SPX up & DXY down", **result}
            if (spx < spx_prev) and (dxy > dxy_prev):
                return {"signal": "bearish", "reason": "SPX down & DXY up", **result}
        # fallback neutral
        return {"signal": "neutral", "reason": "no decisive macro reading", **result}
    except Exception as e:
        print("Macro check error:", e)
        return {"signal":"neutral", "reason":"error or data unavailable"}

# -------------------------
# News headline scan (optional simple)
# -------------------------
def fetch_news_headlines(q="crypto"):
    if not NEWSAPI_KEY:
        return None
    url = ("https://newsapi.org/v2/everything?"
           f"q={q}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWSAPI_KEY}")
    try:
        r = requests.get(url, timeout=10)
        j = r.json()
        if j.get('status') == 'ok':
            headlines = [a['title'] for a in j.get('articles', [])]
            return headlines
    except Exception as e:
        print("News fetch error:", e)
    return None

# -------------------------
# Orchestration
# -------------------------
def evaluate_all():
    now = datetime.now(timezone.utc).astimezone(tz.tzlocal())
    fng = sentiment_check()
    macro = macro_check() if (SPX_TICKER or DXY_TICKER) else {"signal":"neutral","reason":"disabled"}
    results = []
    for sym in SYMBOLS:
        try:
            tech = technical_signal(sym)
            if not tech:
                results.append((sym, None, "technical_fail"))
                continue

            # Need at least technical + (sentiment or macro) aligned in direction
            tech_dir = tech['direction'].lower()
            sentiment_dir = fng['signal'] if fng else "neutral"
            macro_dir = macro['signal'] if macro else "neutral"

            # map to bullish/bearish for comparison
            mapping = {"LONG":"bullish","SHORT":"bearish"}
            tech_bullbear = mapping.get(tech['direction'], "neutral")

            # check confluence
            confluence_count = 0
            if sentiment_dir == tech_bullbear:
                confluence_count += 1
            if macro_dir == tech_bullbear:
                confluence_count += 1

            # requirement: technical mandatory + at least one of the others
            if confluence_count >= 1:
                # compute risk/reward check
                entry = float(tech['entry'])
                sl = float(tech['sl'])
                tp = float(tech['tp'])
                if tech['direction'] == "LONG":
                    risk = entry - sl
                    reward = tp - entry
                else:
                    risk = sl - entry
                    reward = entry - tp
                rr = (reward / risk) if risk > 0 else float('inf')
                if rr < MIN_RR:
                    results.append((sym,None,"rr_too_low"))
                    continue

                # All checks passed -> format signal
                confidence = "High" if rr >= MIN_RR*1.5 and confluence_count == 2 else "Medium"
                signal = {
                    "Asset": sym,
                    "Direction": tech['direction'],
                    "Confidence Level": confidence,
                    "Entry Price": entry,
                    "Take Profit (TP)": tp,
                    "Stop Loss (SL)": sl,
                    "Justification": {
                        "Technical Rationale": (
                            f"Trend: {tech.get('trend')}; entry near current price; invalidation at 1H swing level. "
                            f"Indicators: recent RSI (15m) and Stochastic (15m) aligned."
                        ),
                        "Sentiment/News Driver": f"Fear & Greed: {fng['value']} ({fng['classification']})" if fng else "N/A",
                        "Macroeconomic Context": f"Macro signal: {macro['signal']} ({macro.get('reason')})" if macro else "N/A"
                    },
                    "timestamp_local": now.isoformat()
                }
                results.append((sym, signal, "ok"))
            else:
                results.append((sym,None,"no_confluence"))

        except Exception as e:
            results.append((sym,None,f"error:{e}"))
    return results

def pretty_print(results):
    any_signal = False
    for sym, signal, status in results:
        if status == "ok" and signal:
            any_signal = True
            print_signal(signal)
        else:
            # For debugging: print summarized reason
            print(f"[{sym}] No signal -> reason: {status}")
    if not any_signal:
        print("Market conditions are currently unfavorable. No high-probability trading setups identified for BTC, ETH, or SOL. Standing by for a clearer opportunity.")

def print_signal(s):
    # match user's required output format exactly
    print()
    print(f"Asset: {s['Asset']}")
    print(f"Direction: {s['Direction']}")
    print(f"Confidence Level: {s['Confidence Level']}")
    print(f"Entry Price: {s['Entry Price']}")
    print(f"Take Profit (TP): {s['Take Profit (TP)']}")
    print(f"Stop Loss (SL): {s['Stop Loss (SL)']}")
    print()
    print("Justification:")
    print(f"* Technical Rationale: {s['Justification']['Technical Rationale']}")
    print(f"* Sentiment/News Driver: {s['Justification']['Sentiment/News Driver']}")
    print(f"* Macroeconomic Context: {s['Justification']['Macroeconomic Context']}")
    print(f"[Signal timestamp local]: {s['timestamp_local']}")
    print()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print("Running AI Crypto Analyst evaluation...")
    results = evaluate_all()
    pretty_print(results)
