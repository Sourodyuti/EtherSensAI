"""
Ultimate AI Crypto Analyst & Signal Generator (Enhanced)
Implements comprehensive multi-layered analysis with:
 - Advanced technical analysis with candlestick patterns
 - Multi-timeframe confluence (4H, 1H, 15M)
 - Sentiment analysis (Fear & Greed Index)
 - Macro analysis (SPX/DXY)
 - News headline scanning
 - Automated signal generation with strict validation

Usage:
    python ai_crypto_analyst_enhanced.py
"""

import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import time
from datetime import datetime, timezone
import yfinance as yf
from dateutil import tz
import numpy as np

# -------------------------
# Configuration
# -------------------------
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
EXCHANGE_ID = "binance"

# Timeframes
TF_4H = "4h"
TF_1H = "1h"
TF_15M = "15m"

# Indicator parameters
RSI_PERIOD = 14
STOCH_K = 14
STOCH_D = 3
EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14

# Risk management
MIN_RR = 2.5  # Lowered from 3.0 - accept slightly lower R:R for more opportunities
MAX_SL_DISTANCE_PCT = 0.04  # Lowered from 0.06 (6%) to 0.04 (4%) - tighter stops
MIN_CONFIDENCE_LAYERS = 2  # Require at least 2 of 3 layers (Technical always required)

# Macro tickers
DXY_TICKER = "DX-Y.NYB"
SPX_TICKER = "^GSPC"

# API Keys (optional)
NEWSAPI_KEY = "b84f417086344a90b1f2ec5237b509f7"

# -------------------------
# Initialize Exchange
# -------------------------
exchange = getattr(ccxt, EXCHANGE_ID)({
    'enableRateLimit': True,
})

# -------------------------
# Data Fetching
# -------------------------
def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500):
    """Fetch OHLCV data and return as DataFrame"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching OHLCV for {symbol} {timeframe}: {e}")
        return None

def fetch_current_price(symbol: str):
    """Fetch current market price"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker['last'])
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

# -------------------------
# Technical Indicators
# -------------------------
def add_indicators(df: pd.DataFrame):
    """Add all technical indicators to DataFrame"""
    if df is None or df.empty:
        return None
    
    df = df.copy()
    df['ema50'] = ta.ema(df['close'], length=EMA_FAST)
    df['ema200'] = ta.ema(df['close'], length=EMA_SLOW)
    df['rsi'] = ta.rsi(df['close'], length=RSI_PERIOD)
    
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=STOCH_K, d=STOCH_D)
    if stoch is not None and not stoch.empty:
        df['stoch_k'] = stoch[f'STOCHk_{STOCH_K}_{STOCH_D}_3']
        df['stoch_d'] = stoch[f'STOCHd_{STOCH_K}_{STOCH_D}_3']
    
    atr = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
    if atr is not None:
        df['atr'] = atr
    
    return df

# -------------------------
# Candlestick Pattern Recognition
# -------------------------
def detect_candlestick_patterns(df: pd.DataFrame, lookback=5):
    """
    Detect major candlestick patterns
    Returns: dict with pattern name and signal (bullish/bearish/neutral)
    """
    if df is None or len(df) < lookback:
        return {"pattern": "none", "signal": "neutral", "strength": 0}
    
    recent = df.tail(lookback).copy()
    last = recent.iloc[-1]
    prev = recent.iloc[-2] if len(recent) > 1 else last
    
    o, h, l, c = last['open'], last['high'], last['low'], last['close']
    po, ph, pl, pc = prev['open'], prev['high'], prev['low'], prev['close']
    
    body = abs(c - o)
    prev_body = abs(pc - po)
    range_size = h - l
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    
    patterns = []
    
    # Bullish Patterns
    if c > o:  # Green candle
        # Hammer
        if lower_shadow > 2 * body and upper_shadow < body * 0.3:
            patterns.append(("hammer", "bullish", 2))
        # Bullish Engulfing
        if pc > po and c > po and o < pc and body > prev_body * 1.2:
            patterns.append(("bullish_engulfing", "bullish", 3))
    
    # Bearish Patterns
    if c < o:  # Red candle
        # Shooting Star
        if upper_shadow > 2 * body and lower_shadow < body * 0.3:
            patterns.append(("shooting_star", "bearish", 2))
        # Bearish Engulfing
        if pc < po and c < po and o > pc and body > prev_body * 1.2:
            patterns.append(("bearish_engulfing", "bearish", 3))
    
    # Doji (indecision)
    if body < range_size * 0.1 and range_size > 0:
        patterns.append(("doji", "neutral", 1))
    
    # Morning Star (3-candle bullish reversal)
    if len(recent) >= 3:
        c1, c2, c3 = recent.iloc[-3], recent.iloc[-2], recent.iloc[-1]
        if (c1['close'] < c1['open'] and  # First red
            abs(c2['close'] - c2['open']) < (c1['high'] - c1['low']) * 0.3 and  # Small body
            c3['close'] > c3['open'] and c3['close'] > c1['open']):  # Third green
            patterns.append(("morning_star", "bullish", 3))
    
    # Evening Star (3-candle bearish reversal)
    if len(recent) >= 3:
        c1, c2, c3 = recent.iloc[-3], recent.iloc[-2], recent.iloc[-1]
        if (c1['close'] > c1['open'] and  # First green
            abs(c2['close'] - c2['open']) < (c1['high'] - c1['low']) * 0.3 and  # Small body
            c3['close'] < c3['open'] and c3['close'] < c1['open']):  # Third red
            patterns.append(("evening_star", "bearish", 3))
    
    if not patterns:
        return {"pattern": "none", "signal": "neutral", "strength": 0}
    
    # Return strongest pattern
    patterns.sort(key=lambda x: x[2], reverse=True)
    return {"pattern": patterns[0][0], "signal": patterns[0][1], "strength": patterns[0][2]}

# -------------------------
# Market Structure Analysis
# -------------------------
def analyze_market_structure(df_4h, df_1h):
    """
    Analyze market structure across timeframes
    Returns: trend direction and strength
    """
    if df_4h is None or df_1h is None:
        return {"trend": "neutral", "strength": "weak", "structure": "unclear"}
    
    # 4H trend
    last_4h = df_4h.tail(20)
    ema50_4h = last_4h['ema50'].iloc[-1]
    ema200_4h = last_4h['ema200'].iloc[-1]
    
    # 1H trend
    last_1h = df_1h.tail(20)
    ema50_1h = last_1h['ema50'].iloc[-1]
    ema200_1h = last_1h['ema200'].iloc[-1]
    
    # Determine trend
    trend_4h = "bullish" if ema50_4h > ema200_4h else "bearish" if ema50_4h < ema200_4h else "neutral"
    trend_1h = "bullish" if ema50_1h > ema200_1h else "bearish" if ema50_1h < ema200_1h else "neutral"
    
    # Confluence
    if trend_4h == trend_1h and trend_4h != "neutral":
        strength = "strong"
        trend = trend_4h
    elif trend_4h != "neutral":
        strength = "moderate"
        trend = trend_4h
    else:
        strength = "weak"
        trend = "neutral"
    
    # Structure: higher highs/lows or lower highs/lows
    highs_1h = last_1h['high'].values
    lows_1h = last_1h['low'].values
    
    structure = "unclear"
    if len(highs_1h) >= 3:
        if highs_1h[-1] > highs_1h[-2] > highs_1h[-3] and lows_1h[-1] > lows_1h[-2]:
            structure = "higher_highs_lows"
        elif highs_1h[-1] < highs_1h[-2] < highs_1h[-3] and lows_1h[-1] < lows_1h[-2]:
            structure = "lower_highs_lows"
    
    return {"trend": trend, "strength": strength, "structure": structure}

def find_key_levels(df, lookback=50):
    """Find support and resistance levels"""
    if df is None or len(df) < lookback:
        return {"resistance": None, "support": None}
    
    recent = df.tail(lookback)
    highs = recent['high']
    lows = recent['low']
    
    resistance = float(highs.max())
    support = float(lows.min())
    
    return {"resistance": resistance, "support": support}

def detect_liquidity_sweep(df, lookback=30):
    """Detect potential liquidity sweeps"""
    if df is None or len(df) < lookback:
        return {"swept": False, "direction": None}
    
    recent = df.tail(lookback)
    recent_high = recent['high'].max()
    recent_low = recent['low'].min()
    
    last_candles = df.tail(5)
    last_high = last_candles['high'].max()
    last_low = last_candles['low'].min()
    
    # Check if recent price swept above highs then reversed
    if last_high >= recent_high * 0.999 and last_candles['close'].iloc[-1] < last_high * 0.995:
        return {"swept": True, "direction": "bearish", "level": recent_high}
    
    # Check if recent price swept below lows then reversed
    if last_low <= recent_low * 1.001 and last_candles['close'].iloc[-1] > last_low * 1.005:
        return {"swept": True, "direction": "bullish", "level": recent_low}
    
    return {"swept": False, "direction": None}

# -------------------------
# Technical Analysis (Layer A)
# -------------------------
def technical_analysis(symbol):
    """
    Comprehensive technical analysis
    Returns signal dict or None
    """
    try:
        # Fetch data
        df_4h = add_indicators(fetch_ohlcv(symbol, TF_4H, limit=200))
        df_1h = add_indicators(fetch_ohlcv(symbol, TF_1H, limit=200))
        df_15m = add_indicators(fetch_ohlcv(symbol, TF_15M, limit=200))
        
        if df_4h is None or df_1h is None or df_15m is None:
            return None
        
        # Current price
        price = float(df_15m['close'].iloc[-1])
        
        # Market structure
        structure = analyze_market_structure(df_4h, df_1h)
        
        # Key levels
        levels_1h = find_key_levels(df_1h, lookback=40)
        
        # Liquidity analysis
        liq_sweep = detect_liquidity_sweep(df_1h)
        
        # Candlestick patterns on 15m
        candle_pattern = detect_candlestick_patterns(df_15m, lookback=5)
        
        # Indicators
        rsi_15m = float(df_15m['rsi'].iloc[-1])
        stoch_k_15m = float(df_15m['stoch_k'].iloc[-1])
        stoch_d_15m = float(df_15m['stoch_d'].iloc[-1])
        atr_1h = float(df_1h['atr'].iloc[-1]) if 'atr' in df_1h.columns else price * 0.02
        
        # RSI Divergence detection (simplified)
        rsi_divergence = None
        if len(df_15m) >= 20:
            recent_prices = df_15m['close'].tail(20)
            recent_rsi = df_15m['rsi'].tail(20)
            if recent_prices.iloc[-1] < recent_prices.iloc[-10] and recent_rsi.iloc[-1] > recent_rsi.iloc[-10]:
                rsi_divergence = "bullish"
            elif recent_prices.iloc[-1] > recent_prices.iloc[-10] and recent_rsi.iloc[-1] < recent_rsi.iloc[-10]:
                rsi_divergence = "bearish"
        
        # Signal generation logic
        signal = None
        
        # LONG setup conditions
        if structure['trend'] == "bullish" or (liq_sweep['swept'] and liq_sweep['direction'] == "bullish"):
            # Candlestick confirmation
            candle_bullish = candle_pattern['signal'] == "bullish" and candle_pattern['strength'] >= 2
            
            # RSI not overbought
            rsi_ok = 30 < rsi_15m < 70
            
            # Stochastic turning up or oversold
            stoch_ok = stoch_k_15m < 50 or (stoch_k_15m > stoch_d_15m and stoch_k_15m < 70)
            
            # Divergence bonus
            div_ok = rsi_divergence == "bullish" if rsi_divergence else True
            
            if candle_bullish and rsi_ok and stoch_ok and div_ok:
                # Calculate SL and TP with tighter risk management
                sl = levels_1h['support'] * 0.997 if levels_1h['support'] else price - (atr_1h * 1.5)
                sl = max(sl, price * (1 - MAX_SL_DISTANCE_PCT))
                
                risk = price - sl
                if risk > 0 and risk / price <= MAX_SL_DISTANCE_PCT:
                    tp = price + (risk * MIN_RR)
                    
                    signal = {
                        "direction": "LONG",
                        "entry": round(price, 2),
                        "sl": round(sl, 2),
                        "tp": round(tp, 2),
                        "technical_details": {
                            "trend": structure['trend'],
                            "structure": structure['structure'],
                            "candle_pattern": candle_pattern['pattern'],
                            "rsi": round(rsi_15m, 2),
                            "stoch_k": round(stoch_k_15m, 2),
                            "divergence": rsi_divergence,
                            "liquidity_sweep": liq_sweep['swept']
                        }
                    }
        
        # SHORT setup conditions
        elif structure['trend'] == "bearish" or (liq_sweep['swept'] and liq_sweep['direction'] == "bearish"):
            # Candlestick confirmation
            candle_bearish = candle_pattern['signal'] == "bearish" and candle_pattern['strength'] >= 2
            
            # RSI not oversold
            rsi_ok = 30 < rsi_15m < 70
            
            # Stochastic turning down or overbought
            stoch_ok = stoch_k_15m > 50 or (stoch_k_15m < stoch_d_15m and stoch_k_15m > 30)
            
            # Divergence bonus
            div_ok = rsi_divergence == "bearish" if rsi_divergence else True
            
            if candle_bearish and rsi_ok and stoch_ok and div_ok:
                # Calculate SL and TP with tighter risk management
                sl = levels_1h['resistance'] * 1.003 if levels_1h['resistance'] else price + (atr_1h * 1.5)
                sl = min(sl, price * (1 + MAX_SL_DISTANCE_PCT))
                
                risk = sl - price
                if risk > 0 and risk / price <= MAX_SL_DISTANCE_PCT:
                    tp = price - (risk * MIN_RR)
                    
                    signal = {
                        "direction": "SHORT",
                        "entry": round(price, 2),
                        "sl": round(sl, 2),
                        "tp": round(tp, 2),
                        "technical_details": {
                            "trend": structure['trend'],
                            "structure": structure['structure'],
                            "candle_pattern": candle_pattern['pattern'],
                            "rsi": round(rsi_15m, 2),
                            "stoch_k": round(stoch_k_15m, 2),
                            "divergence": rsi_divergence,
                            "liquidity_sweep": liq_sweep['swept']
                        }
                    }
        
        return signal
        
    except Exception as e:
        print(f"Technical analysis error for {symbol}: {e}")
        return None

# -------------------------
# Sentiment Analysis (Layer B)
# -------------------------
def fetch_fear_and_greed():
    """Fetch Fear & Greed Index from Alternative.me"""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        if r.status_code == 200:
            j = r.json()
            if 'data' in j and len(j['data']) > 0:
                val = int(j['data'][0]['value'])
                classification = j['data'][0]['value_classification']
                return {"value": val, "classification": classification}
    except Exception as e:
        print(f"Fear & Greed fetch error: {e}")
    return None

def fetch_news_headlines():
    """Fetch recent crypto news headlines"""
    if not NEWSAPI_KEY:
        return None
    
    try:
        url = (f"https://newsapi.org/v2/everything?"
               f"q=crypto OR bitcoin OR ethereum&language=en&"
               f"sortBy=publishedAt&pageSize=5&apiKey={NEWSAPI_KEY}")
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            j = r.json()
            if j.get('status') == 'ok':
                articles = j.get('articles', [])
                headlines = [{"title": a['title'], "source": a['source']['name']} 
                            for a in articles[:5]]
                return headlines
    except Exception as e:
        print(f"News fetch error: {e}")
    return None

def sentiment_analysis():
    """
    Analyze market sentiment
    Returns sentiment signal and details
    """
    fng = fetch_fear_and_greed()
    news = fetch_news_headlines()
    
    sentiment_signal = "neutral"
    details = {}
    
    if fng:
        val = fng['value']
        details['fear_greed'] = fng
        
        # Contrarian approach
        if val <= 25:
            sentiment_signal = "bullish"
            details['reason'] = "Extreme fear - contrarian buy signal"
        elif val >= 75:
            sentiment_signal = "bearish"
            details['reason'] = "Extreme greed - contrarian sell signal"
        else:
            details['reason'] = "Neutral sentiment"
    
    if news:
        details['headlines'] = news
        # Simple sentiment from headlines
        positive_keywords = ['rally', 'surge', 'bullish', 'gain', 'approval', 'adoption']
        negative_keywords = ['crash', 'drop', 'bearish', 'fall', 'reject', 'concern']
        
        news_sentiment = 0
        for h in news:
            title_lower = h['title'].lower()
            if any(kw in title_lower for kw in positive_keywords):
                news_sentiment += 1
            if any(kw in title_lower for kw in negative_keywords):
                news_sentiment -= 1
        
        if news_sentiment > 0:
            details['news_bias'] = "positive"
        elif news_sentiment < 0:
            details['news_bias'] = "negative"
        else:
            details['news_bias'] = "neutral"
    
    return {"signal": sentiment_signal, "details": details}

# -------------------------
# Macro Analysis (Layer C)
# -------------------------
def macro_analysis():
    """
    Analyze macroeconomic factors
    Returns macro signal and context
    """
    try:
        context = {}
        signal = "neutral"
        
        # Fetch SPX
        if SPX_TICKER:
            spx_data = yf.Ticker(SPX_TICKER).history(period="5d", interval="1h")
            if not spx_data.empty:
                spx_current = float(spx_data['Close'].iloc[-1])
                spx_prev = float(spx_data['Close'].iloc[-10])
                spx_change = ((spx_current - spx_prev) / spx_prev) * 100
                context['spx'] = {
                    "current": round(spx_current, 2),
                    "change_pct": round(spx_change, 2)
                }
        
        # Fetch DXY
        if DXY_TICKER:
            dxy_data = yf.Ticker(DXY_TICKER).history(period="5d", interval="1h")
            if not dxy_data.empty:
                dxy_current = float(dxy_data['Close'].iloc[-1])
                dxy_prev = float(dxy_data['Close'].iloc[-10])
                dxy_change = ((dxy_current - dxy_prev) / dxy_prev) * 100
                context['dxy'] = {
                    "current": round(dxy_current, 2),
                    "change_pct": round(dxy_change, 2)
                }
        
        # Determine signal
        if 'spx' in context and 'dxy' in context:
            spx_up = context['spx']['change_pct'] > 0.5
            dxy_down = context['dxy']['change_pct'] < -0.3
            spx_down = context['spx']['change_pct'] < -0.5
            dxy_up = context['dxy']['change_pct'] > 0.3
            
            if spx_up and dxy_down:
                signal = "bullish"
                context['reason'] = "Risk-on: SPX rising, DXY falling"
            elif spx_down and dxy_up:
                signal = "bearish"
                context['reason'] = "Risk-off: SPX falling, DXY rising"
            else:
                context['reason'] = "Mixed macro signals"
        else:
            context['reason'] = "Insufficient macro data"
        
        return {"signal": signal, "context": context}
        
    except Exception as e:
        print(f"Macro analysis error: {e}")
        return {"signal": "neutral", "context": {"reason": f"Error: {e}"}}

# -------------------------
# Signal Generator
# -------------------------
def generate_signals():
    """
    Main signal generation orchestrator
    Evaluates all assets and produces signals
    """
    timestamp = datetime.now(timezone.utc).astimezone(tz.tzlocal())
    
    # Fetch sentiment and macro (applies to all assets)
    sentiment = sentiment_analysis()
    macro = macro_analysis()
    
    signals = []
    
    for symbol in SYMBOLS:
        try:
            print(f"\nAnalyzing {symbol}...")
            
            # Technical analysis (mandatory)
            tech = technical_analysis(symbol)
            
            if not tech:
                signals.append({
                    "asset": symbol,
                    "signal": None,
                    "reason": "No valid technical setup"
                })
                continue
            
            # Check confluence with sentiment and macro
            tech_direction = tech['direction'].lower()
            tech_bias = "bullish" if tech_direction == "long" else "bearish"
            
            layers_aligned = 0
            layers_detail = []
            
            # Technical layer (always present)
            layers_aligned += 1
            layers_detail.append("Technical")
            
            # Sentiment layer
            if sentiment['signal'] == tech_bias:
                layers_aligned += 1
                layers_detail.append("Sentiment")
            
            # Macro layer
            if macro['signal'] == tech_bias:
                layers_aligned += 1
                layers_detail.append("Macro")
            
            # Require at least 2 layers (Technical + one other)
            if layers_aligned < MIN_CONFIDENCE_LAYERS:
                signals.append({
                    "asset": symbol,
                    "signal": None,
                    "reason": f"Insufficient confluence ({layers_aligned}/3 layers)"
                })
                continue
            
            # Calculate confidence
            if layers_aligned == 3:
                confidence = "High"
            elif layers_aligned == 2:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Verify risk-reward
            entry = tech['entry']
            sl = tech['sl']
            tp = tech['tp']
            
            if tech_direction == "long":
                risk = entry - sl
                reward = tp - entry
            else:
                risk = sl - entry
                reward = entry - tp
            
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < MIN_RR:
                signals.append({
                    "asset": symbol,
                    "signal": None,
                    "reason": f"Insufficient R:R ({rr_ratio:.2f} < {MIN_RR})"
                })
                continue
            
            # Build technical rationale
            tech_details = tech['technical_details']
            tech_rationale = (
                f"{tech_details['trend'].capitalize()} trend on 4H/1H "
                f"with {tech_details['structure']} structure. "
                f"Candlestick: {tech_details['candle_pattern'].replace('_', ' ').title()} on 15M. "
                f"RSI: {tech_details['rsi']} (confirming momentum). "
                f"Stochastic K: {tech_details['stoch_k']} "
            )
            
            if tech_details['divergence']:
                tech_rationale += f"with {tech_details['divergence']} RSI divergence. "
            
            if tech_details['liquidity_sweep']:
                tech_rationale += "Liquidity sweep detected for potential reversal."
            
            # Build sentiment rationale
            sentiment_rationale = ""
            if 'fear_greed' in sentiment['details']:
                fg = sentiment['details']['fear_greed']
                sentiment_rationale = f"Fear & Greed: {fg['value']} ({fg['classification']})"
                if 'reason' in sentiment['details']:
                    sentiment_rationale += f" - {sentiment['details']['reason']}"
            
            if 'headlines' in sentiment['details'] and sentiment['details']['headlines']:
                sentiment_rationale += f". Recent headlines show {sentiment['details'].get('news_bias', 'mixed')} bias"
            
            # Build macro rationale
            macro_rationale = ""
            if 'spx' in macro['context']:
                spx = macro['context']['spx']
                macro_rationale += f"SPX: {spx['current']} ({spx['change_pct']:+.2f}%). "
            if 'dxy' in macro['context']:
                dxy = macro['context']['dxy']
                macro_rationale += f"DXY: {dxy['current']} ({dxy['change_pct']:+.2f}%). "
            macro_rationale += macro['context'].get('reason', '')
            
            # Create final signal
            final_signal = {
                "asset": symbol,
                "direction": tech['direction'],
                "confidence": confidence,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "rr_ratio": round(rr_ratio, 2),
                "layers_aligned": f"{layers_aligned}/3 ({', '.join(layers_detail)})",
                "justification": {
                    "technical": tech_rationale,
                    "sentiment": sentiment_rationale or "N/A",
                    "macro": macro_rationale or "N/A"
                },
                "timestamp": timestamp.isoformat()
            }
            
            signals.append({
                "asset": symbol,
                "signal": final_signal,
                "reason": "Valid signal generated"
            })
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            signals.append({
                "asset": symbol,
                "signal": None,
                "reason": f"Processing error: {e}"
            })
    
    return signals

# -------------------------
# Output Formatter
# -------------------------
def print_signals(signals):
    """Print signals in the specified format"""
    valid_signals = [s for s in signals if s['signal'] is not None]
    
    if not valid_signals:
        print("\n" + "="*80)
        print("NO SIGNAL PROTOCOL ACTIVATED")
        print("="*80)
        print("\nMarket conditions are currently unfavorable or lack sufficient confluence.")
        print("No high-probability trading setups identified for BTC, ETH, or SOL at this time.")
        print("Monitoring for emerging opportunities, including potential candlestick reversals.")
        print("Standing by for updates.")
        print("\nReasons for no signals:")
        for s in signals:
            print(f"  • {s['asset']}: {s['reason']}")
        print("="*80 + "\n")
        return
    
    print("\n" + "="*80)
    print("AI CRYPTO ANALYST - TRADING SIGNALS")
    print("="*80 + "\n")
    
    for item in valid_signals:
        sig = item['signal']
        
        print(f"Asset: {sig['asset']}")
        print(f"Direction: {sig['direction']}")
        print(f"Confidence Level: {sig['confidence']}")
        print(f"Entry Price: {sig['entry']}")
        print(f"Take Profit (TP): {sig['tp']}")
        print(f"Stop Loss (SL): {sig['sl']}")
        print(f"Risk:Reward Ratio: 1:{sig['rr_ratio']}")
        print(f"Layer Confluence: {sig['layers_aligned']}")
        print()
        print("Justification:")
        print(f"  * Technical Rationale: {sig['justification']['technical']}")
        print(f"  * Sentiment/News Driver: {sig['justification']['sentiment']}")
        print(f"  * Macroeconomic Context: {sig['justification']['macro']}")
        print()
        print(f"[Signal Generated: {sig['timestamp']}]")
        print("-" * 80 + "\n")
    
    # Show rejected signals summary
    rejected = [s for s in signals if s['signal'] is None]
    if rejected:
        print("Assets without valid signals:")
        for s in rejected:
            print(f"  • {s['asset']}: {s['reason']}")
        print()
    
    print("="*80)
    print("DISCLAIMER: These signals are for educational purposes only.")
    print("Always conduct your own research and manage risk appropriately.")
    print("="*80 + "\n")

# -------------------------
# Main Execution
# -------------------------
def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("ULTIMATE AI CRYPTO ANALYST - AUTOMATED MODE")
    print("="*80)
    print(f"\nTimestamp: {datetime.now(timezone.utc).astimezone(tz.tzlocal()).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Assets: {', '.join(SYMBOLS)}")
    print(f"Timeframes: 4H (trend), 1H (structure), 15M (entry)")
    print(f"Min R:R: 1:{MIN_RR} | Max SL: {MAX_SL_DISTANCE_PCT*100}%")
    print(f"Required Confluence: {MIN_CONFIDENCE_LAYERS}/3 layers (Technical always required)")
    print("Risk Profile: CONSERVATIVE (Tight stops, lower leverage suitable)")
    print("\nInitiating multi-layered analysis...\n")
    
    try:
        # Generate signals
        signals = generate_signals()
        
        # Print results
        print_signals(signals)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("Unable to complete analysis. Please check configuration and data sources.\n")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()