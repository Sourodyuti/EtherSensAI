import requests
import pandas as pd
import ta
from datetime import datetime, timezone
from dateutil import tz
import yfinance as yf

# -----------------------------
# Terminal Colors
# -----------------------------
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

# -----------------------------
# Config
# -----------------------------
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SWING_INTERVAL = "1h"
SCALP_INTERVAL = "5m"
MIN_RR_SWING = 3
MIN_RR_SCALP = 1.5
SPX_TICKER = "^GSPC"

# -----------------------------
# Fetch Binance OHLCV
# -----------------------------
def fetch_binance_ohlcv(symbol="BTCUSDT", interval="15m", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume","c1","c2","c3","c4","c5","c6"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df[["time","open","high","low","close","volume"]]

# -----------------------------
# Add Technical Indicators
# -----------------------------
def add_indicators(df):
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bollinger_high"] = bb.bollinger_hband()
    df["bollinger_low"] = bb.bollinger_lband()
    return df

# -----------------------------
# Fear & Greed Index
# -----------------------------
def fetch_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1")
        if r.status_code == 200:
            j = r.json()
            val = int(j['data'][0]['value'])
            classification = j['data'][0]['value_classification']
            return {"value": val, "classification": classification}
    except:
        return None

# -----------------------------
# Macro Signal
# -----------------------------
def macro_signal():
    try:
        if SPX_TICKER:
            spx = yf.Ticker(SPX_TICKER).history(period="5d", interval="1h")['Close']
            spx_now = spx.iloc[-1]
            spx_prev = spx.iloc[-5]
            if spx_now > spx_prev:
                return "bullish"
            elif spx_now < spx_prev:
                return "bearish"
        return "neutral"
    except:
        return "neutral"

# -----------------------------
# Generate Structured Signal
# -----------------------------
def generate_signal(symbol, df, mode="swing"):
    latest = df.iloc[-1]
    price = latest["close"]
    signal = {
        "Asset": symbol,
        "Direction": "NEUTRAL",
        "Confidence Level": "Low",
        "Entry Price": None,
        "Take Profit (TP)": None,
        "Stop Loss (SL)": None,
        "Justification": {
            "Technical Rationale": "",
            "Sentiment/News Driver": "",
            "Macroeconomic Context": ""
        },
        "timestamp_local": datetime.now(timezone.utc).astimezone(tz.tzlocal()).isoformat()
    }

    # --- Sentiment ---
    fng = fetch_fng()
    sentiment_dir = "neutral"
    if fng:
        if fng["value"] <= 25:
            sentiment_dir = "bullish"
        elif fng["value"] >= 75:
            sentiment_dir = "bearish"

    macro_dir = macro_signal()

    # --- Swing Trade ---
    if mode == "swing":
        trend = "bullish" if latest["ema50"] > latest["ema200"] else "bearish"
        valid = False
        if trend == "bullish" and latest["rsi"] < 70:
            signal["Direction"] = "LONG"
            valid = True
        elif trend == "bearish" and latest["rsi"] > 30:
            signal["Direction"] = "SHORT"
            valid = True

        if valid:
            sl = price*0.97 if signal["Direction"]=="LONG" else price*1.03
            tp = price*1.03 if signal["Direction"]=="LONG" else price*0.97
            signal["Entry Price"] = round(price,2)
            signal["Take Profit (TP)"] = round(tp,2)
            signal["Stop Loss (SL)"] = round(sl,2)
            signal["Confidence Level"] = "High" if sentiment_dir==trend or macro_dir==trend else "Medium"
            signal["Justification"]["Technical Rationale"] = f"EMA trend {trend} + RSI confirmation"

    # --- Scalp Trade ---
    elif mode == "scalp":
        valid = False
        if latest["macd"] > latest["macd_signal"] and price < latest["bollinger_high"]:
            signal["Direction"] = "LONG"
            valid = True
        elif latest["macd"] < latest["macd_signal"] and price > latest["bollinger_low"]:
            signal["Direction"] = "SHORT"
            valid = True

        if valid:
            sl = price*0.995 if signal["Direction"]=="LONG" else price*1.005
            tp = price*1.01 if signal["Direction"]=="LONG" else price*0.99
            signal["Entry Price"] = round(price,2)
            signal["Take Profit (TP)"] = round(tp,2)
            signal["Stop Loss (SL)"] = round(sl,2)
            signal["Confidence Level"] = "Medium"
            signal["Justification"]["Technical Rationale"] = "MACD crossover + Bollinger confirmation"

    signal["Justification"]["Sentiment/News Driver"] = f"Fear & Greed Index: {fng['value']} ({fng['classification']})" if fng else "N/A"
    signal["Justification"]["Macroeconomic Context"] = f"Macro signal: {macro_dir}"

    return signal

# -----------------------------
# Pretty Print
# -----------------------------
def print_signal(signal, trade_type="Swing Trade"):
    color = bcolors.YELLOW
    if signal["Direction"]=="LONG":
        color = bcolors.GREEN
    elif signal["Direction"]=="SHORT":
        color = bcolors.RED

    print(f"{color}{trade_type} Signal for {signal['Asset']}{bcolors.RESET}")
    print(f"  Direction       : {signal['Direction']}")
    print(f"  Confidence      : {signal['Confidence Level']}")
    print(f"  Entry Price     : {float(signal['Entry Price']) if signal['Entry Price'] else 'N/A'}")
    print(f"  Take Profit (TP): {float(signal['Take Profit (TP)']) if signal['Take Profit (TP)'] else 'N/A'}")
    print(f"  Stop Loss (SL)  : {float(signal['Stop Loss (SL)']) if signal['Stop Loss (SL)'] else 'N/A'}")
    print(f"  Technical       : {signal['Justification']['Technical Rationale']}")
    print(f"  Sentiment       : {signal['Justification']['Sentiment/News Driver']}")
    print(f"  Macro Context   : {signal['Justification']['Macroeconomic Context']}")
    print(f"  Timestamp       : {signal['timestamp_local']}")
    print("-"*60)

# -----------------------------
# Run All Assets
# -----------------------------
def run_all():
    for asset in ASSETS:
        # Swing
        df_swing = fetch_binance_ohlcv(asset, interval=SWING_INTERVAL)
        df_swing = add_indicators(df_swing)
        swing_signal = generate_signal(asset, df_swing, mode="swing")

        # Scalp
        df_scalp = fetch_binance_ohlcv(asset, interval=SCALP_INTERVAL)
        df_scalp = add_indicators(df_scalp)
        scalp_signal = generate_signal(asset, df_scalp, mode="scalp")

        print(f"\n--- {asset} ---")
        print_signal(swing_signal, "Swing Trade")
        print_signal(scalp_signal, "Scalp Trade")

if __name__=="__main__":
    run_all()
