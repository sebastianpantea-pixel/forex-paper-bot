"""
Forex Intraday Bot — Flask Server
Paper Trading cu date reale de la Yahoo Finance

Perechi: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, XAU/USD
Strategie: Triple EMA Trend Filter + ATR Stop + Session Filter
- Trend pe H1: EMA50 vs EMA200
- Entry pe M15: pullback la EMA21
- ATR-based SL/TP
- Session filter: London + NY only
- Max 1 pozitie per pereche
- Max 4 pozitii totale
"""

import os
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

# ─── Config ───────────────────────────────────────────────────────────────────
PAIRS_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "XAUUSD": "GC=F",
}

# ─── State ────────────────────────────────────────────────────────────────────
state = {
    "running": False,
    "capital": 10000.0,
    "pnl": 0.0,
    "daily_dd": 0.0,
    "wins": 0,
    "losses": 0,
    "positions": [],
    "history": [],
    "log": [],
    "scan_count": 0,
    "last_scan": None,
    "regime": {},
    "trades_today": 0,
    "consecutive_losses": 0,
    "last_trade_ts": {},
    "current_day": datetime.utcnow().strftime("%Y-%m-%d"),
    "session": "closed",
}

settings = {
    "pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "XAUUSD"],
    "capital": 10000.0,
    "risk_pct": 0.5,
    "dd_limit": 3.0,
    "rr_ratio": 2.0,
    "atr_sl_mult": 1.5,
    "atr_period": 14,
    "trend_ema_fast": 50,
    "trend_ema_slow": 200,
    "entry_ema": 21,
    "scan_interval": 300,
    "max_trades_day": 4,
    "max_positions": 4,
    "cooldown_minutes": 60,
    "pause_after_losses": 3,
    "session_filter": True,
}

bot_thread = None
state_lock = threading.Lock()
STATE_FILE = Path(os.environ.get("STATE_FILE", "forex_state.json"))

PERSIST_KEYS = ["positions", "history", "pnl", "wins", "losses",
                "trades_today", "daily_dd", "consecutive_losses",
                "last_trade_ts", "current_day", "capital"]

# ─── Persistenta ──────────────────────────────────────────────────────────────
def save_state():
    try:
        with state_lock:
            payload = {k: state[k] for k in PERSIST_KEYS}
            payload["settings"] = dict(settings)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(STATE_FILE)
    except Exception as e:
        print(f"[WARN] save_state: {e}")


def load_state():
    if not STATE_FILE.exists():
        print("[INFO] Pornire fresh.")
        return
    try:
        payload = json.loads(STATE_FILE.read_text())
        with state_lock:
            for k in PERSIST_KEYS:
                if k in payload:
                    state[k] = payload[k]
            if "settings" in payload:
                settings.update(payload["settings"])
        print(f"[INFO] State restaurat: {len(state['positions'])} pozitii, {len(state['history'])} trades.")
    except Exception as e:
        print(f"[WARN] load_state: {e}")


import atexit
_state_loaded = False

@app.before_request
def _load_on_first():
    global _state_loaded
    if not _state_loaded:
        _state_loaded = True
        load_state()

atexit.register(save_state)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def add_log(msg, level="info"):
    entry = {"time": datetime.utcnow().strftime("%H:%M:%S"), "msg": msg, "level": level}
    with state_lock:
        state["log"].insert(0, entry)
        if len(state["log"]) > 300:
            state["log"] = state["log"][:300]
    print(f"[{entry['time']}] {msg}")


def reset_daily_if_needed():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    do_log = False
    with state_lock:
        if state["current_day"] != today:
            state["current_day"] = today
            state["trades_today"] = 0
            state["daily_dd"] = 0.0
            state["consecutive_losses"] = 0
            do_log = True
    if do_log:
        add_log("Reset zilnic efectuat.", "info")


def get_session():
    """Detecteaza sesiunea forex curenta (UTC)."""
    now_utc = datetime.utcnow()
    h = now_utc.hour
    # London: 08:00-17:00 UTC
    # New York: 13:00-22:00 UTC
    # Overlap (cel mai bun): 13:00-17:00 UTC
    if 13 <= h < 17:
        return "overlap"
    elif 8 <= h < 17:
        return "london"
    elif 13 <= h < 22:
        return "newyork"
    elif 22 <= h or h < 8:
        return "asian"
    return "closed"


def is_tradeable_session():
    if not settings["session_filter"]:
        return True
    session = get_session()
    return session in ("london", "newyork", "overlap")

# ─── Date de piata ─────────────────────────────────────────────────────────────
def get_price(pair):
    """Pret curent pentru o pereche."""
    ticker = PAIRS_MAP.get(pair)
    if not ticker:
        return None
    try:
        data = yf.Ticker(ticker).fast_info
        return float(data.last_price)
    except Exception as e:
        add_log(f"[{pair}] Price error: {e}", "err")
        return None


def get_klines(pair, interval="1h", limit=250):
    """
    Descarca date OHLCV.
    interval: '15m', '1h', '1d'
    """
    ticker = PAIRS_MAP.get(pair)
    if not ticker:
        return pd.DataFrame()
    try:
        period_map = {"15m": "7d", "1h": "60d", "1d": "365d"}
        period = period_map.get(interval, "60d")
        df = yf.download(ticker, interval=interval, period=period,
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.tail(limit)
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        return df
    except Exception as e:
        add_log(f"[{pair}] Klines error: {e}", "err")
        return pd.DataFrame()

# ─── Indicatori ───────────────────────────────────────────────────────────────
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def atr(df, period=14):
    h = df["high"]
    l = df["low"]
    c = df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def detect_regime(df_h1):
    """Trending / ranging bazat pe ATR relativ."""
    if df_h1.empty or len(df_h1) < 20:
        return "unknown"
    a = atr(df_h1, 14).iloc[-1]
    price = df_h1["close"].iloc[-1]
    atr_pct = (a / price) * 100
    if atr_pct > 0.3:
        return "trending"
    if atr_pct < 0.1:
        return "ranging"
    return "normal"

# ─── Strategie ────────────────────────────────────────────────────────────────
def detect_trend(df_h1):
    """
    Trend pe H1: EMA50 vs EMA200.
    Returneaza 'LONG', 'SHORT' sau None.
    """
    if df_h1.empty or len(df_h1) < settings["trend_ema_slow"] + 5:
        return None

    closes = df_h1["close"]
    ema_fast = ema(closes, settings["trend_ema_fast"])
    ema_slow = ema(closes, settings["trend_ema_slow"])

    fast_now  = ema_fast.iloc[-1]
    fast_prev = ema_fast.iloc[-2]
    slow_now  = ema_slow.iloc[-1]
    price     = closes.iloc[-1]

    bullish = fast_now > slow_now and price > fast_now and fast_now > fast_prev
    bearish = fast_now < slow_now and price < fast_now and fast_now < fast_prev

    if bullish:
        return "LONG"
    if bearish:
        return "SHORT"
    return None


def find_entry(df_m15, direction, atr_val):
    """
    Entry pe M15: pullback la EMA21, confirmare candle.
    Returneaza dict cu entry/sl/tp sau None.
    """
    if df_m15.empty or len(df_m15) < 30:
        return None

    closes = df_m15["close"]
    highs  = df_m15["high"]
    lows   = df_m15["low"]
    opens  = df_m15["open"]

    ema21 = ema(closes, settings["entry_ema"])

    # Ultimele 3 candele
    c0 = closes.iloc[-1]   # confirmare
    c1 = closes.iloc[-2]   # pullback
    o0 = opens.iloc[-1]
    o1 = opens.iloc[-2]
    h0 = highs.iloc[-1]
    l0 = lows.iloc[-1]
    h1 = highs.iloc[-2]
    l1 = lows.iloc[-2]
    e0 = ema21.iloc[-1]
    e1 = ema21.iloc[-2]

    sl_dist = atr_val * settings["atr_sl_mult"]

    if direction == "LONG":
        # Pullback: candle bearish aproape de EMA21
        pb_touched = l1 <= e1 * 1.002
        pb_bearish = c1 < o1
        # Confirmare: candle bullish care inchide peste high-ul anterior
        confirm = c0 > o0 and c0 > h1
        # SL sub low-ul pullback-ului
        sl = min(l0, l1) - sl_dist * 0.3
        tp = c0 + (c0 - sl) * settings["rr_ratio"]

        if pb_touched and pb_bearish and confirm:
            return {"direction": "LONG", "entry": c0, "sl": sl, "tp": tp,
                    "reason": f"H1 trend LONG + M15 pullback EMA{settings['entry_ema']}"}

    if direction == "SHORT":
        # Pullback: candle bullish aproape de EMA21
        pb_touched = h1 >= e1 * 0.998
        pb_bullish = c1 > o1
        # Confirmare: candle bearish care inchide sub low-ul anterior
        confirm = c0 < o0 and c0 < l1
        # SL peste high-ul pullback-ului
        sl = max(h0, h1) + sl_dist * 0.3
        tp = c0 - (sl - c0) * settings["rr_ratio"]

        if pb_touched and pb_bullish and confirm:
            return {"direction": "SHORT", "entry": c0, "sl": sl, "tp": tp,
                    "reason": f"H1 trend SHORT + M15 pullback EMA{settings['entry_ema']}"}

    return None


def scan_pair(pair):
    """Scaneaza o pereche si returneaza semnal sau None."""
    df_h1  = get_klines(pair, "1h", 250)
    df_m15 = get_klines(pair, "15m", 100)

    if df_h1.empty or df_m15.empty:
        add_log(f"[{pair}] Date insuficiente.", "warn")
        return None

    regime = detect_regime(df_h1)
    with state_lock:
        state["regime"][pair] = regime

    trend = detect_trend(df_h1)
    add_log(f"[{pair}] Trend:{trend or '—'} | Regim:{regime}")

    if trend is None:
        return None

    atr_val = atr(df_h1, settings["atr_period"]).iloc[-1]
    if pd.isna(atr_val) or atr_val <= 0:
        return None

    signal = find_entry(df_m15, trend, atr_val)
    return signal

# ─── Risk ─────────────────────────────────────────────────────────────────────
def can_trade(pair):
    reset_daily_if_needed()

    with state_lock:
        daily_dd     = state["daily_dd"]
        trades_today = state["trades_today"]
        cons_losses  = state["consecutive_losses"]
        open_all     = [p for p in state["positions"] if p["status"] == "open"]
        open_pair    = [p for p in open_all if p["pair"] == pair]
        last_ts      = state["last_trade_ts"].get(pair, 0)

    if daily_dd >= settings["dd_limit"]:
        add_log(f"[{pair}] Daily DD {daily_dd:.2f}% atins.", "warn")
        return False

    if trades_today >= settings["max_trades_day"]:
        add_log(f"[{pair}] Max trades/zi atins.", "warn")
        return False

    if cons_losses >= settings["pause_after_losses"]:
        add_log(f"[{pair}] Pauza dupa {cons_losses} pierderi consecutive.", "warn")
        return False

    if len(open_all) >= settings["max_positions"]:
        add_log(f"[{pair}] Max {settings['max_positions']} pozitii simultan.", "warn")
        return False

    if open_pair:
        add_log(f"[{pair}] Pozitie deja deschisa.", "warn")
        return False

    cooldown_s = settings["cooldown_minutes"] * 60
    since = time.time() - last_ts
    if last_ts > 0 and since < cooldown_s:
        rem = int((cooldown_s - since) / 60) + 1
        add_log(f"[{pair}] Cooldown ~{rem} min.", "warn")
        return False

    return True

# ─── Executie (paper) ─────────────────────────────────────────────────────────
def open_trade(pair, signal):
    price = get_price(pair)
    if not price:
        price = signal["entry"]

    sl  = float(signal["sl"])
    tp  = float(signal["tp"])
    direction = signal["direction"]

    with state_lock:
        capital = state["capital"] + state["pnl"]

    risk_usd = capital * (settings["risk_pct"] / 100)
    sl_dist  = abs(price - sl)
    if sl_dist <= 0:
        return

    # Units = risc / distanta SL (in termeni de pret)
    units = round(risk_usd / sl_dist, 2)

    pos = {
        "id":         f"{pair}_{int(time.time())}",
        "pair":       pair,
        "direction":  direction,
        "entry":      price,
        "mark_price": price,
        "sl":         sl,
        "tp":         tp,
        "units":      units,
        "status":     "open",
        "open_time":  datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "reason":     signal["reason"],
        "pnl_live":   0.0,
    }

    with state_lock:
        state["positions"].append(pos)
        state["trades_today"] += 1
        state["last_trade_ts"][pair] = time.time()

    add_log(
        f"✅ [PAPER] {direction} {pair} @ {price:.5f} | SL:{sl:.5f} TP:{tp:.5f} | {signal['reason']}",
        "ok"
    )
    save_state()


def update_positions():
    """Actualizeaza P&L live si verifica SL/TP."""
    with state_lock:
        open_positions = [p for p in state["positions"] if p["status"] == "open"]

    for pos in open_positions:
        price = get_price(pos["pair"])
        if not price:
            continue

        if pos["direction"] == "LONG":
            pnl_live = (price - pos["entry"]) * pos["units"]
        else:
            pnl_live = (pos["entry"] - price) * pos["units"]

        with state_lock:
            pos["mark_price"] = price
            pos["pnl_live"]   = round(pnl_live, 2)

        hit_sl = (pos["direction"] == "LONG"  and price <= pos["sl"]) or \
                 (pos["direction"] == "SHORT" and price >= pos["sl"])
        hit_tp = (pos["direction"] == "LONG"  and price >= pos["tp"]) or \
                 (pos["direction"] == "SHORT" and price <= pos["tp"])

        if hit_tp:
            add_log(f"🎯 TP atins {pos['pair']} @ {price:.5f}", "ok")
            close_trade(pos, price, pnl_live, "TP hit")
        elif hit_sl:
            add_log(f"🛑 SL atins {pos['pair']} @ {price:.5f}", "warn")
            close_trade(pos, price, pnl_live, "SL hit")


def close_trade(pos, exit_price, pnl, reason):
    with state_lock:
        pos["status"]     = "closed"
        pos["close_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        pos["exit"]       = exit_price
        pos["pnl_live"]   = round(pnl, 2)
        state["pnl"]     += pnl

        if pnl > 0:
            state["wins"]              += 1
            state["consecutive_losses"] = 0
        else:
            state["losses"]             += 1
            state["consecutive_losses"] += 1
            state["daily_dd"]           += abs(pnl) / max(state["capital"], 1) * 100

    sl_dist = abs(pos["entry"] - pos["sl"])
    rr_val  = abs((exit_price - pos["entry"]) / sl_dist) if sl_dist > 0 else 0
    rr      = f"1:{rr_val:.1f}"

    with state_lock:
        state["history"].append({
            "date":   datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "pair":   pos["pair"],
            "dir":    pos["direction"],
            "entry":  pos["entry"],
            "exit":   exit_price,
            "pnl":    round(pnl, 2),
            "rr":     rr,
            "reason": reason,
        })
        dd_hit = state["daily_dd"] >= settings["dd_limit"]

    add_log(
        f"{'✅' if pnl > 0 else '❌'} {pos['pair']} {reason} | {pnl:+.2f} USD | {rr}",
        "ok" if pnl > 0 else "err"
    )
    save_state()

    if dd_hit:
        with state_lock:
            state["running"] = False
        add_log(f"⛔ Daily DD atins. Bot oprit!", "err")

# ─── Bot Loop ─────────────────────────────────────────────────────────────────
def bot_loop():
    add_log("🟢 Forex Paper Bot pornit.", "ok")
    time.sleep(2)

    while True:
        with state_lock:
            running = state["running"]
        if not running:
            break

        try:
            reset_daily_if_needed()

            session = get_session()
            with state_lock:
                state["session"]    = session
                state["scan_count"] += 1
                state["last_scan"]  = datetime.utcnow().strftime("%H:%M:%S")

            # Actualizeaza pozitii deschise la fiecare scan
            update_positions()

            if not is_tradeable_session():
                add_log(f"Sesiune {session} — nu trade-uim.", "info")
            else:
                add_log(f"Sesiune: {session.upper()} — scan activ.", "info")
                pairs = list(settings.get("pairs", []))
                for pair in pairs:
                    with state_lock:
                        if not state["running"]:
                            break
                    if can_trade(pair):
                        signal = scan_pair(pair)
                        if signal:
                            add_log(f"[{pair}] 📊 {signal['reason']}", "warn")
                            open_trade(pair, signal)
                        else:
                            add_log(f"[{pair}] Niciun setup.", "info")

        except Exception as e:
            add_log(f"Eroare bot_loop: {e}", "err")

        interval = int(settings.get("scan_interval", 300))
        for _ in range(interval):
            with state_lock:
                if not state["running"]:
                    break
            time.sleep(1)

    add_log("⛔ Bot oprit.", "warn")

# ─── API ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def api_status():
    with state_lock:
        total        = state["wins"] + state["losses"]
        capital      = state["capital"]
        pnl          = state["pnl"]
        daily_dd     = state["daily_dd"]
        wins         = state["wins"]
        losses       = state["losses"]
        scan_count   = state["scan_count"]
        last_scan    = state["last_scan"]
        regime       = dict(state["regime"])
        trades_today = state["trades_today"]
        cons_losses  = state["consecutive_losses"]
        active       = len([p for p in state["positions"] if p["status"] == "open"])
        running      = state["running"]
        session      = state["session"]
        live_pnl     = sum(p.get("pnl_live", 0) for p in state["positions"] if p["status"] == "open")

    # Regim agregat
    if regime:
        vals = list(regime.values())
        regime_agg = "trending" if "trending" in vals else (vals[0] if vals else "unknown")
    else:
        regime_agg = "unknown"

    return jsonify({
        "running":           running,
        "paper":             True,
        "capital":           round(capital, 2),
        "pnl":               round(pnl, 2),
        "live_pnl":          round(live_pnl, 2),
        "pnl_pct":           round(pnl / max(capital, 1) * 100, 2),
        "daily_dd":          round(daily_dd, 2),
        "wins":              wins,
        "losses":            losses,
        "win_rate":          round(wins / total * 100, 1) if total > 0 else 0,
        "active_positions":  active,
        "scan_count":        scan_count,
        "last_scan":         last_scan,
        "regime":            regime_agg,
        "regime_per_pair":   regime,
        "trades_today":      trades_today,
        "consecutive_losses": cons_losses,
        "session":           session,
        "settings":          settings,
    })


@app.route("/api/positions")
def api_positions():
    with state_lock:
        return jsonify(state["positions"])


@app.route("/api/history")
def api_history():
    with state_lock:
        return jsonify(list(reversed(state["history"][-50:])))


@app.route("/api/log")
def api_log():
    with state_lock:
        return jsonify(state["log"][:100])


@app.route("/api/start", methods=["POST"])
def api_start():
    global bot_thread
    with state_lock:
        if state["running"]:
            return jsonify({"ok": False, "msg": "Bot deja pornit"})
        state["running"] = True

    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state_lock:
        state["running"] = False
    return jsonify({"ok": True})


@app.route("/api/close-position", methods=["POST"])
def api_close_position():
    data = request.get_json() or {}
    target = data.get("pair")

    with state_lock:
        open_pos = [p for p in state["positions"] if p["status"] == "open"]

    if not open_pos:
        return jsonify({"ok": False, "msg": "Nicio pozitie deschisa"})

    if target:
        lst = [p for p in open_pos if p["pair"] == target]
        if not lst:
            return jsonify({"ok": False, "msg": f"Nicio pozitie pe {target}"})
        pos = lst[0]
    else:
        pos = open_pos[0]

    price = get_price(pos["pair"]) or pos.get("mark_price") or pos["entry"]
    if pos["direction"] == "LONG":
        pnl = (price - pos["entry"]) * pos["units"]
    else:
        pnl = (pos["entry"] - price) * pos["units"]

    close_trade(pos, price, pnl, "Inchidere manuala")
    return jsonify({"ok": True, "msg": f"Pozitia pe {pos['pair']} inchisa"})


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json() or {}

    if "pairs" in data:
        v = data["pairs"]
        if isinstance(v, list):
            settings["pairs"] = [str(s).strip().upper() for s in v if s]

    for k in ["risk_pct", "dd_limit", "rr_ratio", "atr_sl_mult", "atr_period",
              "trend_ema_fast", "trend_ema_slow", "entry_ema", "scan_interval",
              "max_trades_day", "max_positions", "cooldown_minutes",
              "pause_after_losses", "session_filter", "capital"]:
        if k in data:
            settings[k] = data[k]

    if "capital" in data:
        with state_lock:
            state["capital"] = float(data["capital"])

    add_log(f"Setari actualizate. Perechi: {settings['pairs']}", "info")
    save_state()
    return jsonify({"ok": True, "settings": settings})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
