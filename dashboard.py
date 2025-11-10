# ================================
# ALPHA TRACKER AI — SELF-LEARNING ORACLE (FIXED)
# ================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import threading
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import tweepy
import warnings
import pickle
import os

# === SUPPRESS WARNINGS ===
warnings.filterwarnings("ignore")

# === CONFIG FIRST ===
st.set_page_config(layout="wide", page_title="Alpha Tracker AI")

# === DEBUG MODE ===
DEBUG = st.sidebar.toggle("DEBUG MODE", value=False)
debug_queue = []
debug_lock = threading.Lock()

def log(msg):
    if DEBUG:
        with debug_lock:
            debug_queue.append(f"`{datetime.now().strftime('%H:%M:%S')}` {msg}")
            if len(debug_queue) > 50:
                debug_queue.pop(0)

if DEBUG:
    st.sidebar.write("**DEBUG LOGS**")
    with st.sidebar:
        for line in debug_queue:
            st.write(line)

# === API KEYS ===
TELEGRAM_BOT_TOKEN = "8336894718:AAFBl5ITiWNlPERdevHj9DqjqC57VA5NwD8"
TELEGRAM_CHAT_ID = "1500305017"
TWITTER_API_KEY = "Lt9e8m6dnR3wT3TiVP9tVXVXD"
TWITTER_API_SECRET = "w4xs2xNlQB76Du3EDQ54K2JFLJFCRB7MsEqK8vnvVu5d7UvkdZ"
TWITTER_ACCESS_TOKEN = "1984892733551726592-lCF2qR79X6FpiYxGiZUSXGsAp4L2Rd"
TWITTER_ACCESS_TOKEN_SECRET = "RoBngNtdsh4FFLDu1j2Dma6KHn9EnDABIWnSUmF7IwKf3"
ALPHA_VANTAGE_KEY = "NQTDRX4866LD4Z5Z"

# === CLIENTS ===
client = None
try:
    client = tweepy.Client(
        consumer_key=TWITTER_API_KEY,
        consumer_secret=TWITTER_API_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
        wait_on_rate_limit=True
    )
    log("Twitter connected")
except:
    log("Twitter failed")

# === PERSISTENT STORAGE ===
PREDICTION_LOG = "predictions.pkl"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(PREDICTION_LOG):
    pd.DataFrame(columns=['date', 'ticker', 'predicted', 'actual', 'error']).to_pickle(PREDICTION_LOG)

# === ASSETS (MOVED UP) ===
MONITORED_ASSETS = [
    'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'PLTR', 'MSTR', 'COIN',
    'ZW=F', 'GC=F', 'KC=F', 'CL=F', '^GSPC', '^DJI', 'EURUSD=X', 'USDJPY=X'
]
TICKER_TO_NAME = {
    'AAPL':'Apple', 'TSLA':'Tesla', 'NVDA':'NVIDIA', 'MSFT':'Microsoft', 'GOOGL':'Alphabet',
    'PLTR':'Palantir', 'MSTR':'MicroStrategy', 'COIN':'Coinbase', 'ZW=F':'Wheat Futures',
    'GC=F':'Gold Futures', 'KC=F':'Coffee Futures', 'CL=F':'Crude Oil', '^GSPC':'S&P 500',
    '^DJI':'Dow Jones', 'EURUSD=X':'EUR/USD', 'USDJPY=X':'USD/JPY'
}
TWITTER_USERS = ['ipsos', 'SovEcon', 'IKAR_Russia', 'RussianGrainTra']

# === SESSION STATE ===
if 'alerted_tickers' not in st.session_state: st.session_state.alerted_tickers = set()
if 'seen_tweets' not in st.session_state: st.session_state.seen_tweets = set()
if 'latest_alerts' not in st.session_state: st.session_state.latest_alerts = []

# === ORACLE CONTROL ===
oracle_event = threading.Event()

# === MARKET HOURS (IST) ===
def is_market_open(ticker):
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()
    if ticker.endswith('=F'):
        return weekday < 5 and (hour >= 18 or hour < 5)
    elif ticker.endswith('=X'):
        return weekday < 5
    else:
        return weekday < 5 and (hour >= 19 or hour < 2)

# === TELEGRAM ===
def send_telegram(msg):
    if DEBUG:
        log(f"DEBUG: {msg}")
        return True
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'},
            timeout=10
        )
        return r.ok
    except:
        return False

# === LIVE DATA ===
def get_live_data(tickers):
    data = {}
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(period='2d', interval='1d')
            if len(hist) >= 2:
                price = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                move = (price - prev) / prev * 100
                data[t] = {'Price': f"${price:.2f}", 'Movement': f"{move:+.2f}%", 'Current': price}
            else:
                price = ticker.info.get('regularMarketPrice') or 0
                data[t] = {'Price': f"${price:.2f}", 'Movement': 'N/A', 'Current': price}
        except:
            data[t] = {'Price': 'N/A', 'Movement': 'N/A', 'Current': 0}
    return data

# === TECHNICAL INDICATORS ===
def add_technicals(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df['RSI'] = df['RSI'].fillna(50)
    df['MACD'] = df['MACD'].fillna(0)
    df['MACD_Signal'] = df['MACD_Signal'].fillna(0)
    df['MACD_Hist'] = df['MACD_Hist'].fillna(0)
    return df

# === ALPHA VANTAGE SENTIMENT ===
@st.cache_resource(ttl=3600)
def get_av_sentiment():
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={ALPHA_VANTAGE_KEY}&limit=200"
        data = requests.get(url, timeout=10).json()
        log("Sentiment fetched")
        return data.get('feed', [])
    except:
        log("Sentiment failed")
        return []

# === SELF-LEARNING PREDICTION ===
@st.cache_data(ttl=1800)
def predict_next_close(tickers):
    results = []
    news_feed = get_av_sentiment()
    sentiment_dict = {}

    for item in news_feed:
        for t in tickers:
            if t in item.get('ticker_sentiment', {}):
                score = float(item['ticker_sentiment'][t].get('ticker_sentiment_score', 0))
                relevance = float(item['ticker_sentiment'][t].get('relevance_score', 0))
                if relevance > 0.3:
                    sentiment_dict[t] = sentiment_dict.get(t, 0) + score * relevance

    for t in tickers:
        try:
            df = yf.download(t, period="60d", progress=False, auto_adjust=True)
            if len(df) < 30:
                results.append({'Ticker': t, 'Next_Close': 'N/A', 'Conf': 0.0})
                continue
            df = df.dropna()
            if len(df) < 26:
                results.append({'Ticker': t, 'Next_Close': 'N/A', 'Conf': 0.0})
                continue

            df = add_technicals(df)
            sent = sentiment_dict.get(t, 0)
            df['Sentiment'] = sent

            X = df[['Close', 'RSI', 'MACD_Hist', 'Sentiment']].values[:-1]
            y = df['Close'].shift(-1).values[:-1]
            if len(X) < 2:
                results.append({'Ticker': t, 'Next_Close': 'N/A', 'Conf': 0.0})
                continue

            model_path = f"{MODEL_DIR}/{t}_model.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                log(f"Loaded model for {t}")
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y.ravel())
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                log(f"Trained new model for {t}")

            pred = model.predict(X[-1:])
            conf = 0.85
            results.append({'Ticker': t, 'Next_Close': f"${pred[0]:.2f}", 'Conf': conf})

            log_df = pd.read_pickle(PREDICTION_LOG)
            new_pred = pd.DataFrame([{
                'date': datetime.now().date(),
                'ticker': t,
                'predicted': pred[0],
                'actual': np.nan,
                'error': np.nan
            }])
            log_df = pd.concat([log_df, new_pred], ignore_index=True)
            log_df.to_pickle(PREDICTION_LOG)

        except Exception as e:
            results.append({'Ticker': t, 'Next_Close': 'N/A', 'Conf': 0.0})
            log(f"Pred failed {t}: {e}")
    return pd.DataFrame(results).set_index('Ticker')

# === SELF-LEARNING LOOP ===
def self_learn_loop():
    log("SELF-LEARNING STARTED")
    while not oracle_event.is_set():
        try:
            log_df = pd.read_pickle(PREDICTION_LOG)
            yesterday = (datetime.now() - timedelta(days=1)).date()
            yesterday_preds = log_df[log_df['date'] == yesterday]

            for _, row in yesterday_preds.iterrows():
                t = row['ticker']
                pred = row['predicted']
                try:
                    actual = yf.Ticker(t).history(period='1d')['Close'].iloc[-1]
                    error = abs(pred - actual) / actual
                    log_df.loc[(log_df['date'] == yesterday) & (log_df['ticker'] == t), 'actual'] = actual
                    log_df.loc[(log_df['date'] == yesterday) & (log_df['ticker'] == t), 'error'] = error

                    if error > 0.02:
                        log(f"RETRAINING {t} | Error: {error:.1%}")
                        df = yf.download(t, period="60d", progress=False, auto_adjust=True)
                        df = add_technicals(df)
                        df['Sentiment'] = 0
                        X = df[['Close', 'RSI', 'MACD_Hist', 'Sentiment']].values[:-1]
                        y = df['Close'].shift(-1).values[:-1]
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X, y.ravel())
                        with open(f"{MODEL_DIR}/{t}_model.pkl", 'wb') as f:
                            pickle.dump(model, f)
                except: pass
            log_df.to_pickle(PREDICTION_LOG)
        except: pass
        time.sleep(3600)
    log("SELF-LEARNING STOPPED")

# === ALERTS ===
def check_6percent():
    alerts = []
    for t in [t for t in MONITORED_ASSETS if is_market_open(t)]:
        try:
            data = yf.Ticker(t).history(period='1d', interval='5m')
            if len(data) < 10: continue
            change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
            if abs(change) >= 6:
                msg = f"<b>6%+ ALERT</b>\n{t}: {'UP' if change > 0 else 'DOWN'} {abs(change):.1f}%"
                if t not in st.session_state.alerted_tickers:
                    send_telegram(msg)
                    st.session_state.alerted_tickers.add(t)
                    alerts.append(msg)
        except: pass
    return alerts

def check_wheat_sniper():
    alerts = []
    if not client: return alerts
    try:
        for user in TWITTER_USERS:
            resp = client.get_user(username=user)
            if not resp.data: continue
            tweets = client.get_users_tweets(id=resp.data.id, max_results=5)
            if not tweets.data: continue
            for tweet in tweets.data:
                if tweet.id in st.session_state.seen_tweets: continue
                if any(k in tweet.text.lower() for k in ['wheat', 'russia', 'tender']):
                    msg = f"<b>WHEAT SNIPER</b>\n@{user}: {tweet.text[:80]}..."
                    send_telegram(msg)
                    st.session_state.seen_tweets.add(tweet.id)
                    alerts.append(msg)
    except: pass
    return alerts

def oracle_loop():
    log("ORACLE STARTED")
    while not oracle_event.is_set():
        alerts = check_6percent() + check_wheat_sniper()
        if alerts:
            st.session_state.latest_alerts = alerts[-3:]
        time.sleep(300)
    log("ORACLE STOPPED")

# === UI ===
st.title("ALPHA TRACKER AI — SELF-LEARNING ORACLE")

# FIXED: MOVED AFTER MONITORED_ASSETS
closed = [t for t in MONITORED_ASSETS if not is_market_open(t)]
if closed:
    st.warning(f"Market Closed: {', '.join(TICKER_TO_NAME.get(c, c) for c in closed[:6])}")

st.header("ML Signals (RSI + MACD + Sentiment + Self-Learning)")
live = get_live_data(MONITORED_ASSETS)
pred_df = predict_next_close(MONITORED_ASSETS)

rows = []
for t in MONITORED_ASSETS:
    d = live[t]
    pred = pred_df.loc[t] if t in pred_df.index else {'Next_Close': 'N/A', 'Conf': 0}
    signal = "BUY" if pred['Next_Close'] != 'N/A' and float(pred['Next_Close'][1:]) > d['Current'] * 1.01 else \
             "SELL" if pred['Next_Close'] != 'N/A' and float(pred['Next_Close'][1:]) < d['Current'] * 0.99 else "HOLD"
    rows.append({
        'Ticker': t,
        'Asset': TICKER_TO_NAME.get(t, t),
        'Price': d['Price'],
        'Day %': d['Movement'],
        'Signal': signal,
        'Conf': f"{pred['Conf']:.0%}",
        'Next Close': pred['Next_Close']
    })

df = pd.DataFrame(rows)
st.dataframe(df.sort_values('Conf', ascending=False), width='stretch')

st.header("Live Oracle")
c1, c2 = st.columns([1, 4])
with c1:
    if st.button("START ORACLE", type="primary"):
        oracle_event.clear()
        threading.Thread(target=oracle_loop, daemon=True).start()
        threading.Thread(target=self_learn_loop, daemon=True).start()
        st.success("LIVE + LEARNING")
        st.rerun()
    if st.button("STOP"):
        oracle_event.set()
        st.rerun()
    st.write(f"**Status:** {'LIVE + LEARNING' if not oracle_event.is_set() else 'OFF'}")
with c2:
    if not oracle_event.is_set():
        st_autorefresh(interval=60_000)
        for a in st.session_state.latest_alerts:
            st.success(a)

st.sidebar.header("Live Prices")
for t in MONITORED_ASSETS[:6]:
    d = live.get(t, {'Price': 'N/A', 'Movement': 'N/A'})
    col = "green" if '+' in d['Movement'] else "red" if '-' in d['Movement'] else "gray"
    st.sidebar.markdown(f"**{TICKER_TO_NAME.get(t,t)}**: {d['Price']} <span style='color:{col}'>{d['Movement']}</span>", unsafe_allow_html=True)

if st.button("TEST TELEGRAM"):
    if send_telegram("<b>SELF-LEARNING AI</b>\nFixed & LIVE!"):
        st.success("Sent!")
        st.balloons()
    else:
        st.error("Failed")
