# run_strategy_V8_Reliable_Data_SMC.py
# Nobitex AI Scalper Pro — Reliable Data + Precision SMC Edition
# Technologies: Smart Money Concepts, LSTM/XGBoost/LightGBM Ensemble, 
#               VPIN Microstructure, Multi-Timeframe Alignment, 
#               Kelly Criterion Risk Management, Walk-Forward Backtesting

from __future__ import annotations
import base64
import os
import sys
import time
import json
import csv
import logging
import threading
import traceback
import re
import subprocess
import math
import random
import warnings
import sqlite3
import hashlib
import shutil
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from urllib.parse import urljoin
from typing import Any, Dict, List, Tuple, Optional, Callable

from pathlib import Path

# Early application data paths: must exist before Deep Research persistence is initialized.
APP_DATA_DIR = Path.home() / "Nobitex_AI_Trader_Pro_Data"
CACHE_DATA_DIR = APP_DATA_DIR / "cache"
ML_DATA_DIR = APP_DATA_DIR / "ml"
BACKTEST_DATA_DIR = APP_DATA_DIR / "backtests"
SIGNAL_SCREENSHOT_DIR = APP_DATA_DIR / "signal_screenshots"
for _early_d in (APP_DATA_DIR, CACHE_DATA_DIR, ML_DATA_DIR, BACKTEST_DATA_DIR, SIGNAL_SCREENSHOT_DIR):
    try:
        _early_d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

warnings.filterwarnings("ignore")

# optional libs
try:
    import pyperclip
except Exception:
    pyperclip = None

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# Advanced ML libs with graceful degradation
_ML_AVAILABLE = {}
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import classification_report, r2_score, roc_auc_score, precision_recall_curve
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.decomposition import PCA
    _ML_AVAILABLE["sklearn"] = True
except Exception as e:
    _ML_AVAILABLE["sklearn"] = False
    logging.debug("sklearn not available: %s", e)

try:
    import xgboost as xgb
    _ML_AVAILABLE["xgboost"] = True
except Exception:
    _ML_AVAILABLE["xgboost"] = False

try:
    import lightgbm as lgb
    _ML_AVAILABLE["lightgbm"] = True
except Exception:
    _ML_AVAILABLE["lightgbm"] = False

# TensorFlow DISABLED due to AVX2/CPU compatibility issues on this machine.
# To re-enable after fixing MSVC redist or installing tensorflow-cpu==2.10.1,
# uncomment the block below.
_ML_AVAILABLE["tensorflow"] = False
# try:
#     import tensorflow as tf
#     from tensorflow.keras.models import Sequential, load_model, Model
#     from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Attention, LayerNormalization, Concatenate
#     from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
#     from tensorflow.keras.optimizers import Adam
#     tf.get_logger().setLevel("ERROR")
#     _ML_AVAILABLE["tensorflow"] = True
# except Exception:
#     _ML_AVAILABLE["tensorflow"] = False

try:
    import shap
    _ML_AVAILABLE["shap"] = True
except Exception:
    _ML_AVAILABLE["shap"] = False



# ---------------- Deep Analysis Engine ----------------
DEEP_LOOKBACK_DAYS = 730
DEEP_TIMEFRAMES = (5, 15, 60, 240, 1440)
DEEP_MAX_BARS = 12000
DEEP_CHUNK_BARS = 500

def _deep_df_from_udf(j):
    try:
        if not isinstance(j, dict):
            return pd.DataFrame()
        o = j.get("o") or j.get("open") or []
        h = j.get("h") or j.get("high") or []
        l = j.get("l") or j.get("low") or []
        c = j.get("c") or j.get("close") or []
        v = j.get("v") or j.get("volume") or []
        t = j.get("t") or j.get("time") or []
        n = min(len(o), len(h), len(l), len(c), len(v))
        if n < 10:
            return pd.DataFrame()
        d = pd.DataFrame({
            "open": o[-n:], "high": h[-n:], "low": l[-n:],
            "close": c[-n:], "volume": v[-n:]
        })
        for c0 in d.columns:
            d[c0] = pd.to_numeric(d[c0], errors="coerce")
        d = d.dropna()
        if t:
            tt=t[-len(d):]
            d.index=pd.to_datetime(tt, unit="s")
        else:
            d.index=pd.date_range(end=pd.Timestamp.now(), periods=len(d), freq=f"{DEEP_TIMEFRAMES[0]}min")
        return d[~d.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.DataFrame()

def _deep_fetch_history(sym, resolution, days=DEEP_LOOKBACK_DAYS):
    end=int(time.time())
    start=end-int(days*86400)
    step=int(DEEP_CHUNK_BARS*resolution*60)
    chunks=[]
    cur=start
    guard=0
    candidates=[globals().get("canonical_to_api",{}).get(canonical(sym)), *_candidate_symbol_forms(sym)]
    seen=set()
    candidates=[x for x in candidates if x and not (x in seen or seen.add(x))][:3]
    while cur < end and guard < 500:
        to=min(end, cur+step)
        got=False
        for cs in candidates:
            try:
                j=robust_history_udf(cs, resolution, cur, to, valid_symbols=set(valid_symbols_map.keys()) if valid_symbols_map else None)
                # robust_history_udf may fall back; accept only usable data.
                d=_deep_df_from_udf(j)
                if not d.empty:
                    chunks.append(d)
                    got=True
                    break
            except Exception:
                pass
        cur=to
        guard+=1
    if not chunks:
        return pd.DataFrame()
    d=pd.concat(chunks)
    d=d[~d.index.duplicated(keep="last")].sort_index()
    return d.tail(DEEP_MAX_BARS)

def _deep_candle_patterns(df):
    if len(df)<3: return []
    a=df.iloc[-1]; b=df.iloc[-2]
    out=[]
    body=abs(float(a.close-a.open))
    rng=max(float(a.high-a.low),1e-12)
    upper=float(a.high-max(a.open,a.close))
    lower=float(min(a.open,a.close)-a.low)
    if body/rng<0.12: out.append("دوجی")
    if lower>body*2 and upper<body*1.2: out.append("پین‌بار صعودی")
    if upper>body*2 and lower<body*1.2: out.append("پین‌بار نزولی")
    if b.close<b.open and a.close>a.open and a.close>=b.open and a.open<=b.close:
        out.append("پوشای صعودی")
    if b.close>b.open and a.close<a.open and a.open>=b.close and a.close<=b.open:
        out.append("پوشای نزولی")
    return out

def _deep_levels(df, n=160):
    d=df.tail(min(n,len(df)))
    if d.empty: return [],[]
    sup=[]; res=[]
    for i in range(2,len(d)-2):
        lo=float(d.low.iloc[i]); hi=float(d.high.iloc[i])
        if lo<=float(d.low.iloc[i-1]) and lo<=float(d.low.iloc[i+1]):
            sup.append(lo)
        if hi>=float(d.high.iloc[i-1]) and hi>=float(d.high.iloc[i+1]):
            res.append(hi)
    def cluster(vals):
        vals=sorted(vals); out=[]
        for v in vals:
            if not out or abs(v-out[-1])/max(abs(out[-1]),1e-12)>0.004:
                out.append(v)
            else:
                out[-1]=(out[-1]+v)/2
        return out[-8:]
    return cluster(sup),cluster(res)

def _deep_method_regime(df):
    if len(df)<100:return 50.0
    c=df.close
    e20=c.ewm(span=20,adjust=False).mean().iloc[-1]
    e50=c.ewm(span=50,adjust=False).mean().iloc[-1]
    e100=c.ewm(span=100,adjust=False).mean().iloc[-1]
    score=50
    if c.iloc[-1]>e20>e50>e100: score=88
    elif c.iloc[-1]>e20>e50: score=74
    elif c.iloc[-1]<e20<e50<e100: score=12
    elif c.iloc[-1]<e20<e50: score=26
    ret=(float(c.iloc[-1])/float(c.iloc[-30])-1)*100
    return float(np.clip(score+np.tanh(ret/8)*18,0,100))

def _deep_method_liquidity(df):
    d=df.tail(40)
    rng=(d.high-d.low).replace(0,np.nan)
    pos=((d.close-d.low)/rng).fillna(.5)
    vm=d.volume.rolling(20).mean().replace(0,np.nan)
    vr=(d.volume/vm).replace([np.inf,-np.inf],np.nan).fillna(1)
    x=((pos-.5)*2*vr).clip(-3,3).tail(12).mean()
    return float(np.clip(50+x*16,0,100))

def _deep_method_curvature(df):
    c=df.close
    if len(c)<60:return 50.0
    r1=c.pct_change(5).rolling(5).mean().iloc[-1]
    r2=c.pct_change(5).rolling(5).mean().iloc[-6]
    accel=(float(r1)-float(r2))*1000
    return float(np.clip(50+np.tanh(accel)*25,0,100))

def deep_metrics(df):
    if df is None or df.empty or len(df)<80:return None
    d=df.copy()
    c=d.close
    rsi=calculate_rsi(d); macd=calculate_macd(d); adx=calculate_adx(d); dem=calculate_demarker(d)
    atr=calculate_atr(d,14)
    ub,mid,lb=calculate_bollinger_bands(d)
    vwap=calculate_vwap_daily(d)
    if vwap is None:
        den=float(d.volume.sum())
        if den>0:
            tp=(d.high+d.low+d.close)/3
            vwap=float((tp*d.volume).sum()/den)
    ret30=(float(c.iloc[-1])/float(c.iloc[-31])-1)*100 if len(c)>31 else 0
    ret90=(float(c.iloc[-1])/float(c.iloc[-91])-1)*100 if len(c)>91 else 0
    vm20=float(d.volume.tail(20).mean())
    vm60=float(d.volume.tail(60).mean())
    vol_ratio=vm20/vm60 if vm60 else 1
    regime=_deep_method_regime(d)
    liquidity=_deep_method_liquidity(d)
    curvature=_deep_method_curvature(d)
    ind=50
    if rsi<35: ind+=18
    elif rsi>65: ind-=18
    if macd>0: ind+=12
    elif macd<0: ind-=12
    if dem<.35: ind+=8
    elif dem>.65: ind-=8
    trend=50
    e20=c.ewm(span=20,adjust=False).mean().iloc[-1]
    e50=c.ewm(span=50,adjust=False).mean().iloc[-1]
    if c.iloc[-1]>e20>e50: trend=80
    elif c.iloc[-1]<e20<e50: trend=20
    momentum=float(np.clip(50+(ret30+0.5*ret90)*1.2,0,100))
    supports,resistances=_deep_levels(d)
    combined=(
        trend*.18+momentum*.14+ind*.18+
        regime*.15+liquidity*.17+curvature*.18
    )
    return {
        "bars":len(d),"rsi":rsi,"macd":macd,"adx":adx,"dem":dem,"atr":atr,
        "vwap":vwap,"ret30":ret30,"ret90":ret90,"vol_ratio":vol_ratio,
        "trend":trend,"momentum":momentum,"ind":ind,
        "regime":regime,"liquidity":liquidity,"curvature":curvature,
        "support":supports,"resistance":resistances,
        "patterns":_deep_candle_patterns(d),"score":float(np.clip(combined,0,100))
    }


# ---------------- Independent Method Research ----------------
DEEP_METHOD_NAMES = ("Trend", "Momentum", "Volatility", "روش اختصاصی ۱: رژیم", "روش اختصاصی ۲: نقدینگی", "روش اختصاصی ۳: خمیدگی")


# ---------------- Persistent research checkpoints ----------------
# Every completed method/timeframe is persisted atomically.  A later run resumes
# from the last completed unit instead of repeating expensive two-year research.
# Bump DEEP_RESEARCH_SCHEMA_VERSION whenever the research *structure* changes
# (method definition, parameter grid, split/labeling logic, etc.).
DEEP_RESEARCH_SCHEMA_VERSION = "2026-09-24-R3"

# User-selectable persistent research storage.  The program asks for a folder
# and a master research filename the first time a long research job is started,
# then remembers that choice.  The pointer itself remains in APP_DATA_DIR so
# Windows shutdown/restart cannot lose the selected location.
DEEP_STORAGE_CONFIG = Path(APP_DATA_DIR) / "deep_research_storage.json"
DEEP_RESEARCH_DIR = Path(APP_DATA_DIR) / "deep_method_research"
DEEP_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
DEEP_RESEARCH_MASTER_FILE = DEEP_RESEARCH_DIR / "research_master.json"
DEEP_RESEARCH_BACKUP_DIR = DEEP_RESEARCH_DIR / "backups"
DEEP_RESEARCH_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
DEEP_RESEARCH_BACKUP_KEEP = 8

def _deep_load_storage_config():
    try:
        obj=json.loads(DEEP_STORAGE_CONFIG.read_text(encoding="utf-8"))
        folder=obj.get("folder")
        master=obj.get("master_file")
        if folder and master:
            return {"folder":str(Path(folder)),"master_file":str(Path(master))}
    except Exception:
        pass
    return {}

def _deep_save_storage_config(folder, master_file):
    try:
        DEEP_STORAGE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        tmp=DEEP_STORAGE_CONFIG.with_suffix(".tmp")
        tmp.write_text(json.dumps({"folder":str(folder),"master_file":str(master_file),"updated_at":time.time()},ensure_ascii=False,indent=2),encoding="utf-8")
        os.replace(str(tmp),str(DEEP_STORAGE_CONFIG))
    except Exception as exc:
        logging.debug("deep storage config save failed: %s",exc)

def _deep_set_storage_location(folder, master_file):
    global DEEP_RESEARCH_DIR, DEEP_RESEARCH_BACKUP_DIR, DEEP_RESEARCH_MASTER_FILE
    folder=Path(folder).expanduser().resolve()
    master_file=Path(master_file).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    DEEP_RESEARCH_DIR=folder
    DEEP_RESEARCH_BACKUP_DIR=folder / "backups"
    DEEP_RESEARCH_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    DEEP_RESEARCH_MASTER_FILE=master_file
    _deep_save_storage_config(folder,master_file)
    return folder, master_file

def _deep_choose_storage_location(parent=None, force=False):
    """Ask user for the research folder and master file, then persist the choice."""
    cfg=_deep_load_storage_config()
    if cfg and not force:
        try:
            folder=Path(cfg["folder"]); master=Path(cfg["master_file"])
            if folder.exists():
                return _deep_set_storage_location(folder,master)
        except Exception: pass
    try:
        folder=filedialog.askdirectory(parent=parent,title="انتخاب پوشه ذخیره‌سازی تحقیقات ۲ ساله")
        if not folder:
            return None,None
        default=str(Path(folder)/"deep_research_master.json")
        master=filedialog.asksaveasfilename(parent=parent,title="انتخاب نام فایل اصلی تحقیقات",initialdir=folder,initialfile=Path(default).name,defaultextension=".json",filetypes=[("Research JSON","*.json"),("All files","*.*")])
        if not master:
            return None,None
        return _deep_set_storage_location(folder,master)
    except Exception as exc:
        logging.debug("deep storage chooser failed: %s",exc)
        return None,None

def _deep_ensure_storage_location(parent=None):
    folder,master=_deep_choose_storage_location(parent=parent,force=False)
    if folder and master:
        return True
    # If the user cancels, keep the safe application-local default rather than
    # breaking the scanner.  The next explicit research start will ask again.
    return False

# Restore the user's last selected location on program start.
try:
    _cfg=_deep_load_storage_config()
    if _cfg and Path(_cfg.get("folder","")).exists():
        _deep_set_storage_location(_cfg["folder"],_cfg["master_file"])
except Exception:
    pass

def _deep_write_master_index(extra=None):
    """Write a small human-readable index at the user-selected master path."""
    try:
        payload={"schema":DEEP_RESEARCH_SCHEMA_VERSION,"research_folder":str(DEEP_RESEARCH_DIR),"backup_folder":str(DEEP_RESEARCH_BACKUP_DIR),"updated_at":time.time()}
        if extra: payload.update(extra)
        target=Path(DEEP_RESEARCH_MASTER_FILE); target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix(target.suffix+".tmp")
        tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
        os.replace(str(tmp),str(target))
    except Exception as exc:
        logging.debug("research master index write failed: %s",exc)

def _deep_backup_file(path):
    """Atomically preserve the currently-valid research file before replacing it."""
    try:
        src = Path(path)
        if not src.exists() or not src.is_file():
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000:06d}"
        dest = DEEP_RESEARCH_BACKUP_DIR / f"{src.name}.{stamp}.bak"
        shutil.copy2(str(src), str(dest))
        # Keep only the newest N backups for this exact source file.
        matches = sorted(DEEP_RESEARCH_BACKUP_DIR.glob(src.name + ".*.bak"), key=lambda x: x.stat().st_mtime, reverse=True)
        for old in matches[DEEP_RESEARCH_BACKUP_KEEP:]:
            try: old.unlink()
            except Exception: pass
        return dest
    except Exception as exc:
        logging.debug("research backup failed for %s: %s", path, exc)
        return None

def _deep_backup_all_research():
    """Create a timestamped safety snapshot of all current research artifacts."""
    try:
        stamp = time.strftime("snapshot_%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000:06d}"
        dest = DEEP_RESEARCH_BACKUP_DIR / stamp
        dest.mkdir(parents=True, exist_ok=True)
        count = 0
        for src in DEEP_RESEARCH_DIR.glob("*"):
            if src.name == "backups" or not src.is_file():
                continue
            shutil.copy2(str(src), str(dest / src.name))
            count += 1
        return dest, count
    except Exception as exc:
        logging.debug("research snapshot failed: %s", exc)
        return None, 0

def _deep_restore_latest_research_backup():
    """Restore the latest per-file backup. Returns (restored_count, message)."""
    restored = 0
    try:
        latest_by_name = {}
        for src in DEEP_RESEARCH_BACKUP_DIR.glob("*.bak"):
            base = src.name.rsplit(".", 2)[0]
            if base not in latest_by_name or src.stat().st_mtime > latest_by_name[base].stat().st_mtime:
                latest_by_name[base] = src
        for base, bak in latest_by_name.items():
            target = DEEP_RESEARCH_DIR / base
            tmp = target.with_suffix(target.suffix + ".restore.tmp")
            shutil.copy2(str(bak), str(tmp))
            os.replace(str(tmp), str(target))
            restored += 1
        return restored, f"{restored} فایل تحقیق از آخرین پشتیبان بازیابی شد"
    except Exception as exc:
        return restored, f"بازیابی ناقص: {exc}"

def _deep_research_fingerprint(years=2):
    payload = {
        "schema": DEEP_RESEARCH_SCHEMA_VERSION,
        "methods": list(DEEP_METHOD_NAMES),
        "timeframes": list(DEEP_TIMEFRAMES),
        "years": years,
        "split": "70/30",
        "horizon": 12,
        "grids": {m: _deep_method_grid(m) for m in DEEP_METHOD_NAMES} if "_deep_method_grid" in globals() else {},
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]

def _deep_checkpoint_path(sym, tf):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", canonical(sym))
    return DEEP_RESEARCH_DIR / f"{safe}_{int(tf)}m.checkpoint.json"

def _deep_load_checkpoint(sym, tf, years=2):
    p = _deep_checkpoint_path(sym, tf)
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("fingerprint") != _deep_research_fingerprint(years):
            return {"status":"stale", "methods":{}, "path":str(p)}
        return obj
    except Exception:
        return {"status":"new", "methods":{}, "path":str(p)}

def _deep_save_checkpoint(sym, tf, state, years=2):
    p = _deep_checkpoint_path(sym, tf)
    state = dict(state)
    state.update({
        "symbol": canonical(sym), "timeframe": int(tf), "years": years,
        "fingerprint": _deep_research_fingerprint(years),
        "schema": DEEP_RESEARCH_SCHEMA_VERSION,
        "updated_at": time.time(),
    })
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        # Preserve the last valid checkpoint before replacing it.
        _deep_backup_file(p)
        tmp.write_text(json.dumps(state, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(p))
    except Exception:
        try:
            if tmp.exists(): tmp.unlink()
        except Exception:
            pass
    return p

def _deep_signal_method(df, method, params):
    if df is None or len(df) < 80:
        return 0
    c=df["close"].astype(float)
    if method == "Trend":
        fast=int(params.get("fast",20)); slow=int(params.get("slow",50)); adx_min=float(params.get("adx",18))
        ef=c.ewm(span=fast,adjust=False).mean(); es=c.ewm(span=slow,adjust=False).mean()
        try: adx=float(calculate_adx(df))
        except Exception: adx=0.0
        if adx < adx_min: return 0
        return 1 if c.iloc[-1]>ef.iloc[-1]>es.iloc[-1] else -1 if c.iloc[-1]<ef.iloc[-1]<es.iloc[-1] else 0
    if method == "Momentum":
        rp=int(params.get("rsi",14)); os=float(params.get("os",35)); ob=float(params.get("ob",65))
        rsi=float(calculate_rsi(df, period=rp)) if len(df)>rp+2 else 50.0
        macd=float(calculate_macd(df))
        if rsi<=os and macd>0: return 1
        if rsi>=ob and macd<0: return -1
        return 1 if rsi<45 and macd>0 else -1 if rsi>55 and macd<0 else 0
    if method == "Volatility":
        n=int(params.get("window",20)); k=float(params.get("k",2.0))
        ma=c.rolling(n).mean().iloc[-1]; sd=c.rolling(n).std().iloc[-1]
        if not np.isfinite(ma) or not np.isfinite(sd) or sd<=0: return 0
        if c.iloc[-1] > ma+k*sd: return 1
        if c.iloc[-1] < ma-k*sd: return -1
        return 0
    t=float(params.get("threshold",60)); buy=t; sell=100-t
    if method.startswith("روش اختصاصی ۱"):
        x=_deep_method_regime(df); return 1 if x>=buy else -1 if x<=sell else 0
    if method.startswith("روش اختصاصی ۲"):
        x=_deep_method_liquidity(df); return 1 if x>=buy else -1 if x<=sell else 0
    if method.startswith("روش اختصاصی ۳"):
        x=_deep_method_curvature(df); return 1 if x>=buy else -1 if x<=sell else 0
    return 0

def _deep_forward_outcome(df, i, direction, horizon=12, atr_mult=1.0):
    if direction==0 or i>=len(df)-1: return None
    entry=float(df.close.iloc[i]); atrs=calculate_atr(df.iloc[:i+1],14)
    atr=float(atrs) if np.isfinite(atrs) and atrs>0 else entry*0.01
    tp=entry+direction*atr*atr_mult; sl=entry-direction*atr*atr_mult
    end=min(len(df)-1,i+horizon)
    for j in range(i+1,end+1):
        hi=float(df.high.iloc[j]); lo=float(df.low.iloc[j])
        if direction>0:
            if lo<=sl and hi>=tp: return -1
            if lo<=sl: return -1
            if hi>=tp: return 1
        else:
            if hi>=sl and lo<=tp: return -1
            if hi>=sl: return -1
            if lo<=tp: return 1
    final=float(df.close.iloc[end])
    return 1 if (final-entry)*direction >= 0 else -1

def _deep_eval_method(df, method, params, start, end, horizon=12):
    trades=wins=losses=0; gross_win=gross_loss=0.0; equity=0.0; peak=0.0; dd=0.0
    for i in range(max(start,80), min(end,len(df)-horizon-1)):
        sig=_deep_signal_method(df.iloc[:i+1],method,params)
        if sig==0: continue
        outcome=_deep_forward_outcome(df,i,sig,horizon=horizon)
        if outcome is None: continue
        trades+=1
        ret=1.0 if outcome>0 else -1.0
        equity+=ret; peak=max(peak,equity); dd=max(dd,peak-equity)
        if outcome>0: wins+=1; gross_win+=1.0
        else: losses+=1; gross_loss+=1.0
    wr=(wins/trades*100) if trades else 0.0
    pf=(gross_win/gross_loss) if gross_loss else (gross_win if gross_win else 0.0)
    expectancy=(wins-losses)/trades if trades else 0.0
    score=float(np.clip(50 + expectancy*35 + min(pf,4)*6 - min(dd,20)*0.5,0,100)) if trades else 0.0
    return {"trades":trades,"wins":wins,"losses":losses,"win_rate":wr,"profit_factor":pf,"expectancy":expectancy,"max_drawdown":dd,"score":score,"params":dict(params)}

def _deep_method_grid(method):
    if method=="Trend":
        return [{"fast":f,"slow":s,"adx":a} for f in (10,20,30) for s in (50,100) if f<s for a in (15,20,25)]
    if method=="Momentum":
        return [{"rsi":r,"os":os,"ob":ob} for r in (10,14,21) for os,ob in ((30,70),(35,65),(40,60))]
    if method=="Volatility":
        return [{"window":w,"k":k} for w in (14,20,30) for k in (1.5,2.0,2.5)]
    return [{"threshold":t} for t in (55,60,65,70)]

def _deep_research_method(df, method):
    if df is None or len(df)<250: return {"status":"insufficient","method":method}
    cut=int(len(df)*0.70); grid=_deep_method_grid(method)
    train=[]
    for p0 in grid:
        train.append(_deep_eval_method(df,method,p0,80,cut))
    train=[x for x in train if x["trades"]>=5]
    if not train: return {"status":"no_trades","method":method}
    train.sort(key=lambda x:(x["score"],x["profit_factor"],x["win_rate"]),reverse=True)
    candidates=train[:min(5,len(train))]
    oos=[]
    for cand in candidates:
        e=_deep_eval_method(df,method,cand["params"],cut,len(df)-1)
        oos.append(e)
    best=max(oos,key=lambda x:(x["score"],x["profit_factor"],x["win_rate"])) if oos else candidates[0]
    return {"status":"ok","method":method,"train":candidates,"best_train":candidates[0],"oos":oos,"best":best}

def _deep_research_combination(df, research):
    winners={m:r.get("best") for m,r in research.items() if r.get("status")=="ok" and r.get("best")}
    if len(winners)<2 or df is None or len(df)<250: return {"status":"insufficient","methods":len(winners)}
    cut=int(len(df)*0.70); trades=wins=losses=0; equity=peak=dd=0.0
    for i in range(max(cut,80),len(df)-13):
        votes=[]
        for m,b in winners.items():
            votes.append(_deep_signal_method(df.iloc[:i+1],m,b["params"]))
        vote=sum(votes)
        sig=1 if vote>=2 else -1 if vote<=-2 else 0
        if not sig: continue
        out=_deep_forward_outcome(df,i,sig,horizon=12)
        if out is None: continue
        trades+=1; wins += out>0; losses += out<0
        equity += 1 if out>0 else -1; peak=max(peak,equity); dd=max(dd,peak-equity)
    wr=wins/trades*100 if trades else 0.0; pf=wins/losses if losses else (wins if wins else 0.0)
    score=float(np.clip(50+(wr-50)*0.7+min(pf,4)*5-dd*0.5,0,100)) if trades else 0.0
    return {"status":"ok","trades":trades,"wins":wins,"losses":losses,"win_rate":wr,"profit_factor":pf,"max_drawdown":dd,"score":score,"methods":list(winners)}

def deep_research_dataframe(df, checkpoint=None, checkpoint_save=None):
    result={}
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    saved = checkpoint.get("methods", {}) if isinstance(checkpoint.get("methods", {}), dict) else {}
    for method in DEEP_METHOD_NAMES:
        # Resume completed methods.  If the checkpoint was written under a
        # different research fingerprint it was already discarded by loader.
        cached = saved.get(method)
        if cached and cached.get("status") in ("ok", "insufficient", "no_trades", "error"):
            result[method] = cached
            if checkpoint_save:
                checkpoint_save(method, cached, result)
            continue
        try:
            result[method]=_deep_research_method(df,method)
        except Exception as e:
            result[method]={"status":"error","method":method,"error":str(e)}
        if checkpoint_save:
            checkpoint_save(method, result[method], result)
    result["COMBINATION"]=_deep_research_combination(df,result)
    if checkpoint_save:
        checkpoint_save("COMBINATION", result["COMBINATION"], result)
    return result

def deep_analyze_symbol(sym):
    result={"symbol":canonical(sym),"timeframes":{},"score":0.0,"best_tf":None,"method_research":{}}
    total=len(DEEP_TIMEFRAMES)
    try:
        if app and hasattr(app, "set_tab_activity"):
            app.set_tab_activity("آزمایشگاه تحلیل عمیق", "در حال تحلیل", "active")
    except Exception: pass
    for pos,tf in enumerate(DEEP_TIMEFRAMES,1):
        try:
            d=_deep_fetch_history(sym,tf)
            if app and hasattr(app, "set_tab_activity"):
                app.set_tab_activity("آزمایشگاه تحلیل عمیق", f"{sym} | TF {tf} | دریافت داده {pos}/{total}", "active")
            m=deep_metrics(d)
            if m:
                try:
                    # The requested research order is preserved: each method is
                    # optimized independently on 70% and then tested OOS on 30%;
                    # only after that is the combination evaluated.
                    
                    cp = _deep_load_checkpoint(sym, tf, years=2)
                    def _save_method(_method, _payload, _all):
                        cp["methods"] = dict(_all)
                        cp["last_completed"] = _method
                        cp["status"] = "running"
                        _deep_save_checkpoint(sym, tf, cp, years=2)
                    m["method_research"] = deep_research_dataframe(d, checkpoint=cp, checkpoint_save=_save_method)
                    cp["methods"] = dict(m["method_research"])
                    cp["status"] = "done"
                    cp["completed_at"] = time.time()
                    _deep_save_checkpoint(sym, tf, cp, years=2)
                except Exception as e: m["method_research"]={"error":str(e)}
                result["timeframes"][tf]=m
            if app and hasattr(app, "set_tab_activity"):
                app.set_tab_activity("آزمایشگاه تحلیل عمیق", f"{sym} | TF {tf} | تکمیل {pos}/{total}", "active")
        except Exception as e:
            logging.debug("deep tf failed %s/%s: %s",sym,tf,e)
    try:
        if app and hasattr(app, "set_tab_activity"):
            app.set_tab_activity("آزمایشگاه تحلیل عمیق", f"{sym} | تکمیل شد", "done")
    except Exception: pass
    if result["timeframes"]:
        weights={5:.10,15:.15,60:.25,240:.25,1440:.25}
        den=sum(weights.get(k,.1) for k in result["timeframes"])
        result["score"]=sum(result["timeframes"][k]["score"]*weights.get(k,.1) for k in result["timeframes"])/max(den,1e-9)
        result["best_tf"]=max(result["timeframes"], key=lambda k:result["timeframes"][k]["score"])
    return result


# ============================================================
# UNIFIED AI TRADER ENGINE
# ============================================================

AI_MIN_SCORE = 62.0
AI_MIN_SCORE_STRONG = 78.0
AI_PRIMARY_THRESHOLD = 65.0
AI_SECONDARY_THRESHOLD = 72.0
AI_RISK_PER_TRADE = 0.75  # percent of equity in paper/live risk model
AI_MAX_OPEN_POSITIONS = 100
# Multi timeframe optimizer settings
OPTIMIZER_ENABLED = True
OPTIMIZER_LOOKBACK_DAYS = 730
OPTIMIZER_TIMEFRAMES = [5,15,30,60,240,1440]
OPTIMIZER_CACHE_FILE = str(BACKTEST_DATA_DIR / "indicator_optimizer.json")

def _safe_pct(current, previous, default=0.0):
    """Safely calculate percentage change, avoiding division-by-zero/NaN crashes."""
    try:
        cur = float(current)
        prev = float(previous)
        if not math.isfinite(cur) or not math.isfinite(prev) or abs(prev) < 1e-12:
            return float(default)
        return (cur / prev - 1.0) * 100.0
    except (TypeError, ValueError, ZeroDivisionError):
        return float(default)


def _ai_clip(x, lo=0.0, hi=100.0):
    try:
        return float(np.clip(float(x), lo, hi))
    except Exception:
        return lo

def _ai_last_price(sym):
    try:
        p = safe_float(valid_symbols_map.get(canonical(sym)))
        if p > 0:
            return p
    except Exception:
        pass
    try:
        p = get_price_from_orderbook(sym, valid_symbols_map)
        return float(p or 0)
    except Exception:
        return 0.0

def _ai_price_action_score(df):
    if df is None or df.empty or len(df) < 40:
        return 50.0, "خنثی", []
    d = df.tail(60)
    c = d["close"]
    ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    last = d.iloc[-1]
    candle = "صعودی" if float(last["close"]) >= float(last["open"]) else "نزولی"
    score = 50.0
    reasons = []

    if c.iloc[-1] > ema20 > ema50:
        score += 28
        reasons.append("روند قیمت بالای EMA20/EMA50")
    elif c.iloc[-1] < ema20 < ema50:
        score -= 28
        reasons.append("روند قیمت زیر EMA20/EMA50")
    elif c.iloc[-1] > ema20:
        score += 12
    elif c.iloc[-1] < ema20:
        score -= 12

    body = abs(float(last["close"] - last["open"]))
    rng = max(float(last["high"] - last["low"]), 1e-12)
    if body / rng < 0.12:
        reasons.append("دوجی / تردید")

    if candle == "صعودی" and float(last["close"]) > ema20:
        score += 7
        reasons.append("کندل صعودی تاییدی")
    elif candle == "نزولی" and float(last["close"]) < ema20:
        score -= 7
        reasons.append("کندل نزولی تاییدی")

    return _ai_clip(score, 0, 100), candle, reasons

def _ai_volume_score(df):
    if df is None or len(df) < 40:
        return 50.0, 1.0
    recent = float(df["volume"].tail(6).mean())
    base = float(df["volume"].tail(48).mean())
    ratio = recent / base if base > 0 else 1.0
    score = 50.0
    if ratio >= 2.0:
        score = 88.0
    elif ratio >= 1.5:
        score = 78.0
    elif ratio >= 1.2:
        score = 67.0
    elif ratio <= 0.7:
        score = 38.0
    elif ratio <= 0.85:
        score = 44.0
    return score, ratio

def _deep_support_resistance(df, lookback=160):
    """Compatibility wrapper for the AI layer; the canonical implementation is _deep_levels."""
    return _deep_levels(df, n=lookback)


def _ai_levels(df):
    supports, resistances = _deep_support_resistance(df, lookback=min(180, len(df)))
    price = float(df["close"].iloc[-1])
    support = max([x for x in supports if x < price] or [float(df["low"].tail(80).min())])
    resistance = min([x for x in resistances if x > price] or [float(df["high"].tail(80).max())])
    return supports, resistances, support, resistance

def _ai_order_block_pressure(df):
    if df is None or len(df) < 30:
        return 50.0, "خنثی", None
    d = df.tail(30).copy()
    vol_mean = float(d["volume"].tail(20).mean())
    candidates = []
    for i in range(max(1, len(d)-12), len(d)-1):
        row = d.iloc[i]
        vr = float(row["volume"]) / vol_mean if vol_mean > 0 else 1.0
        body = float(row["close"] - row["open"])
        if vr >= 1.35:
            candidates.append((abs(body), i, body, vr, float(row["low"]), float(row["high"])))
    if not candidates:
        return 50.0, "خنثی", None

    _, _, body, vr, low, high = max(candidates, key=lambda x: x[0])
    if body > 0:
        return 76.0, "تقاضای قوی / Order Block صعودی", (low, high)
    if body < 0:
        return 24.0, "عرضه قوی / Order Block نزولی", (low, high)
    return 50.0, "خنثی", (low, high)

def _ai_method_regime(df):
    if df is None or len(df) < 80:
        return 50.0, "نامشخص"
    c = df["close"]
    ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    atr = calculate_atr(df, 14)
    p = float(c.iloc[-1])
    atr_pct = abs(float(atr) / max(p, 1e-12)) * 100
    ret = _safe_pct(float(c.iloc[-1]), float(c.iloc[-25]))
    if p > ema20 > ema50 and ret > 1:
        return _ai_clip(78 + ret * 2 - max(0, atr_pct-4)*2), "روند صعودی"
    if p < ema20 < ema50 and ret < -1:
        return _ai_clip(22 + ret * 2 + max(0, atr_pct-4)*2), "روند نزولی"
    if atr_pct > 6:
        return 50.0, "نوسانی پرشتاب"
    return 50.0, "رنج / خنثی"

def _ai_method_liquidity(df):
    s, r, support, resistance = _ai_levels(df)
    p = float(df["close"].iloc[-1])
    if support > 0 and p > support:
        dist_support = abs(p-support)/p*100
    else:
        dist_support = 99
    if resistance > 0 and p < resistance:
        dist_res = abs(resistance-p)/p*100
    else:
        dist_res = 99
    # Near support = buy-side opportunity; near resistance = sell-side pressure.
    if dist_support < 1.2 and dist_support < dist_res:
        return 78.0, "نزدیک حمایت"
    if dist_res < 1.2 and dist_res < dist_support:
        return 22.0, "نزدیک مقاومت"
    return 50.0, "وسط محدوده"

def _ai_method_curvature(df):
    if df is None or len(df) < 50:
        return 50.0
    c = df["close"]
    a = c.pct_change(5).rolling(5).mean().iloc[-1]
    b = c.pct_change(5).rolling(5).mean().iloc[-6]
    acc = (float(a) - float(b)) * 1000
    return _ai_clip(50 + np.tanh(acc) * 32)


def _series_safe_last(series, default=0.0):
    try:
        if series is None or len(series)==0 or pd.isna(series.iloc[-1]): return default
        return float(series.iloc[-1])
    except Exception:
        return default

def calculate_indicator_suite(df):
    """Compact multi-indicator confluence engine inspired by the supplied TradingView/2026 sources."""
    result={}
    if df is None or len(df)<60:
        return result
    c=df["close"].astype(float); h=df["high"].astype(float); l=df["low"].astype(float); v=df["volume"].astype(float)
    ema9=c.ewm(span=9,adjust=False).mean(); ema21=c.ewm(span=21,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean(); sma50=c.rolling(50).mean(); sma200=c.rolling(min(200,len(c))).mean()
    rsi_s=compute_rsi_series(df,14)
    # MACD + signal + histogram
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); macd_s=e12-e26; sig_s=macd_s.ewm(span=9,adjust=False).mean(); hist=macd_s-sig_s
    # Bollinger
    bbmid=c.rolling(20).mean(); bbsd=c.rolling(20).std(); bbup=bbmid+2*bbsd; bblow=bbmid-2*bbsd; bbwidth=(bbup-bblow)/bbmid.replace(0,np.nan)*100
    # stochastic
    lo14=l.rolling(14).min(); hi14=h.rolling(14).max(); k=100*(c-lo14)/(hi14-lo14+1e-12); d=k.rolling(3).mean()
    # Williams %R
    will=-100*(hi14-c)/(hi14-lo14+1e-12)
    # CCI
    tp=(h+l+c)/3; tpma=tp.rolling(20).mean(); mad=(tp-tpma).abs().rolling(20).mean(); cci=(tp-tpma)/(0.015*mad.replace(0,np.nan))
    # MFI
    pos=np.where(tp.diff()>0,tp*v,0.0); neg=np.where(tp.diff()<0,tp*v,0.0); pm=pd.Series(pos,index=df.index).rolling(14).sum(); nm=pd.Series(neg,index=df.index).rolling(14).sum(); mfi=100-(100/(1+pm/(nm+1e-12)))
    # OBV
    obv=(np.sign(c.diff()).fillna(0)*v).cumsum(); obv_ema=obv.ewm(span=20,adjust=False).mean()
    # Momentum 10
    momentum=c-c.shift(10)
    # ATR / ADX / VWAP / channels already supplied by file
    atr=float(calculate_atr(df,14)); adx=float(calculate_adx(df,14))
    bb_squeeze=(_series_safe_last(bbwidth,99.0) <= float(bbwidth.tail(20).quantile(0.2))) if bbwidth.notna().any() else False
    vwap=None
    try:
        den=float(v.tail(50).sum()); vwap=float(((tp*v).tail(50).sum())/den) if den>0 else None
    except Exception: pass
    ich=calculate_ichimoku(df)
    stoch_k=_series_safe_last(k,50); stoch_d=_series_safe_last(d,50)
    result.update({
      "ema9":_series_safe_last(ema9),"ema21":_series_safe_last(ema21),"ema50":_series_safe_last(ema50),"sma50":_series_safe_last(sma50),"sma200":_series_safe_last(sma200),
      "rsi":_series_safe_last(rsi_s,50),"macd":_series_safe_last(macd_s),"macd_signal":_series_safe_last(sig_s),"macd_hist":_series_safe_last(hist),
      "bb_upper":_series_safe_last(bbup),"bb_mid":_series_safe_last(bbmid),"bb_lower":_series_safe_last(bblow),"bb_width":_series_safe_last(bbwidth),"bb_squeeze":bool(bb_squeeze),
      "stoch_k":stoch_k,"stoch_d":stoch_d,"willr":_series_safe_last(will,-50),"cci":_series_safe_last(cci),"mfi":_series_safe_last(mfi,50),
      "obv":_series_safe_last(obv),"obv_ema":_series_safe_last(obv_ema),"momentum":_series_safe_last(momentum),"atr":atr,"atr_pct":atr/max(float(c.iloc[-1]),1e-12)*100,
      "adx":adx,"vwap":vwap,"ichimoku":ich,
    })
    # Generic confluence votes in [-100,100].
    p=float(c.iloc[-1]); score=0.0; votes=[]
    if p>_series_safe_last(ema21): score+=10; votes.append('EMA21↑')
    else: score-=10; votes.append('EMA21↓')
    if _series_safe_last(ema21)>_series_safe_last(ema50): score+=10; votes.append('EMA21>EMA50')
    else: score-=10; votes.append('EMA21<EMA50')
    if _series_safe_last(sma50)>0 and p>_series_safe_last(sma50): score+=6
    else: score-=6
    r=result['rsi']; score += 12 if r>55 else 8 if r>50 else -12 if r<45 else -8 if r<50 else 0
    mh=result['macd_hist']; score += 12 if mh>0 else -12 if mh<0 else 0
    if result['bb_squeeze']: score += 2; votes.append('Bollinger Squeeze')
    score += 7 if stoch_k>stoch_d and stoch_k<80 else -7 if stoch_k<stoch_d and stoch_k>20 else 0
    score += 5 if result['cci']>0 else -5 if result['cci']<0 else 0
    score += 5 if result['mfi']>50 else -5 if result['mfi']<50 else 0
    score += 5 if result['obv']>result['obv_ema'] else -5
    if vwap: score += 5 if p>vwap else -5
    result['confluence_score']=float(max(-100,min(100,score))); result['votes']=votes
    return result


def smart_ai_history(sym: str, preferred_tf: int = 15, bars: int = 220):
    """Choose the best available TF with bounded attempts and a transparent fallback."""
    key = canonical(sym)
    candidates = [int(preferred_tf)]
    for tf in (5, 15, 60, 240):
        if tf not in candidates:
            candidates.append(tf)

    best = pd.DataFrame()
    used_tf = None

    # At most 3 TFs per scan; avoids 5x request amplification on failed markets.
    for tf in candidates[:3]:
        try:
            df = get_candles_cached(key, tf, n=bars, valid_symbols_map=valid_symbols_map)
            if df is None or df.empty:
                continue
            df = df.copy().dropna(subset=["close", "high", "low", "open", "volume"], how="any")
            if len(df) >= 30 and len(df) > len(best):
                best, used_tf = df, tf
                if len(df) >= 60:
                    break
        except Exception as exc:
            logging.debug("smart_ai_history %s tf=%s: %s", key, tf, exc)

    # Price snapshot is intentionally marked as snapshot, not historical data.
    if best.empty:
        try:
            p = float(valid_symbols_map.get(key) or 0)
            if p > 0:
                idx = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="min")
                best = pd.DataFrame(
                    {"open": p, "high": p, "low": p, "close": p, "volume": 0.0},
                    index=idx,
                )
                used_tf = 0
        except Exception:
            pass
    return best, used_tf


def safe_metric_value(value, fallback=float("nan")):
    try:
        x = float(value)
        return x if math.isfinite(x) else fallback
    except Exception:
        return fallback


def fmt_metric(value, digits=2):
    try:
        x = float(value)
        return f"{x:.{digits}f}" if math.isfinite(x) else "—"
    except Exception:
        return "—"

def unified_ai_decision(sym, timeframe=15):
    key = canonical(sym)
    df, used_tf = smart_ai_history(key, timeframe, bars=220)

    # Always return a row with truthful data status. Do not manufacture 0/50 values.
    if df is None or df.empty:
        p = safe_metric_value(valid_symbols_map.get(key), 0.0)
        return {
            "symbol": key,
            "decision": "WAIT",
            "score": 0.0,
            "confidence": 0.0,
            "signal_state": "WAIT",
            "price": p,
            "entry": None,
            "stop_loss": None,
            "tp1": None,
            "tp2": None,
            "risk_reward": 0.0,
            "regime": "نامشخص",
            "data_status": "قیمت موجود؛ History در دسترس نیست",
            "history_ok": False,
            "timeframe_used": None,
        }

    p = float(df["close"].iloc[-1])

    if not used_tf:
        return {
            "symbol": key,
            "decision": "WAIT",
            "score": 0.0,
            "confidence": 0.0,
            "signal_state": "WAIT",
            "price": p,
            "entry": None,
            "stop_loss": None,
            "tp1": None,
            "tp2": None,
            "risk_reward": 0.0,
            "regime": "نامشخص",
            "data_status": "فقط قیمت لحظه‌ای؛ کندل دریافت نشد",
            "history_ok": False,
            "timeframe_used": 0,
        }

    rsi = float(calculate_rsi(df))
    macd = float(calculate_macd(df))
    adx = float(calculate_adx(df))
    dem = float(calculate_demarker(df))
    indx = calculate_indicator_suite(df)
    suite_confluence_bonus = float(indx.get("confluence_score", 0.0)) if indx else 0.0

    pa_score, candle, pa_reasons = _ai_price_action_score(df)
    vol_score, vol_ratio = _ai_volume_score(df)
    ob_score, ob_text, ob_zone = _ai_order_block_pressure(df)
    regime_score, regime = _ai_method_regime(df)
    liquidity_score, liquidity_state = _ai_method_liquidity(df)
    curvature_score = _ai_method_curvature(df)

    # Indicator agreement is intentionally separated from the 3 custom methods.
    ind = 50.0
    if rsi < 35:
        ind += 15
    elif rsi < 45:
        ind += 7
    elif rsi > 65:
        ind -= 15
    elif rsi > 55:
        ind -= 7
    if macd > 0:
        ind += 12
    elif macd < 0:
        ind -= 12
    if adx >= 25:
        ind += 8 if macd != 0 else 0
    if dem < 0.35:
        ind += 5
    elif dem > 0.65:
        ind -= 5
    ind = _ai_clip(ind + suite_confluence_bonus * 0.10)

    mtf_scores = []
    for tf in (5, 15, 60, 240):
        try:
            d2 = get_candles_cached(key, tf, n=100, valid_symbols_map=valid_symbols_map)
            if d2.empty or len(d2) < 40:
                continue
            trend = _ai_method_regime(d2)[0]
            mtf_scores.append(trend)
        except Exception:
            continue
    mtf = sum(mtf_scores)/len(mtf_scores) if mtf_scores else 50.0

    # Weighted consensus.
    score = (
        pa_score * 0.18 +
        ind * 0.18 +
        vol_score * 0.10 +
        ob_score * 0.10 +
        regime_score * 0.12 +
        liquidity_score * 0.10 +
        curvature_score * 0.10 +
        mtf * 0.12
    )
    score = _ai_clip(score)

    # Determine state with primary/secondary confirmation.
    primary = score >= AI_PRIMARY_THRESHOLD or score <= (100-AI_PRIMARY_THRESHOLD)
    secondary = (
        (pa_score >= 65 and ind >= 60 and vol_score >= 60) or
        (pa_score <= 35 and ind <= 40 and vol_score >= 60) or
        (ob_score >= 70 and liquidity_score >= 65) or
        (ob_score <= 30 and liquidity_score <= 35)
    )

    if score >= AI_MIN_SCORE_STRONG and primary and secondary:
        decision = "BUY++"
    elif score >= AI_MIN_SCORE and primary:
        decision = "BUY"
    elif score <= (100-AI_MIN_SCORE_STRONG) and primary and secondary:
        decision = "SELL++"
    elif score <= (100-AI_MIN_SCORE) and primary:
        decision = "SELL"
    else:
        decision = "WAIT"

    supports, resistances, support, resistance = _ai_levels(df)
    extended_patterns = detect_extended_candlestick_patterns(df)
    eq_levels = detect_equal_high_low(df)
    imbalance_score, bullish_fvg, bearish_fvg = detect_market_imbalance_score(df)
    swings_h, swings_l = detect_swing_points(df, left=2, right=2)
    bos = detect_bos_choch(df, swings_h, swings_l)
    if any(x in extended_patterns for x in BULL_CANDLE_NAMES): score = _ai_clip(score + 4)
    if any(x in extended_patterns for x in BEAR_CANDLE_NAMES): score = _ai_clip(score - 4)
    score = _ai_clip(score + imbalance_score*.06)
    if bos.get("choch_bullish"): score = _ai_clip(score + 6)
    if bos.get("choch_bearish"): score = _ai_clip(score - 6)

    primary = score >= AI_PRIMARY_THRESHOLD or score <= (100-AI_PRIMARY_THRESHOLD)
    secondary = ((pa_score>=65 and ind>=60 and vol_score>=60) or
                 (pa_score<=35 and ind<=40 and vol_score>=60) or
                 (ob_score>=70 and liquidity_score>=65) or
                 (ob_score<=30 and liquidity_score<=35))
    if score >= AI_MIN_SCORE_STRONG and primary and secondary: decision="BUY++"
    elif score >= AI_MIN_SCORE and primary: decision="BUY"
    elif score <= (100-AI_MIN_SCORE_STRONG) and primary and secondary: decision="SELL++"
    elif score <= (100-AI_MIN_SCORE) and primary: decision="SELL"
    else: decision="WAIT"

    plan = build_smart_trade_plan(df, decision, p, confidence)
    atr = float(calculate_atr(df, 14))
    stop_distance = max(atr * 1.6, p * 0.003)
    if decision.startswith("BUY"):
        sl = p - stop_distance
        tp1 = p + stop_distance * 1.4
        tp2 = p + stop_distance * 2.2
    elif decision.startswith("SELL"):
        sl = p + stop_distance
        tp1 = p - stop_distance * 1.4
        tp2 = p - stop_distance * 2.2
    else:
        sl = tp1 = tp2 = None

    rr = (
        abs(tp2-p) / max(abs(p-sl), 1e-12)
        if sl and tp2 else 0.0
    )

    confidence = _ai_clip(50 + abs(score-50)*1.5)
    signal_state = (
        "🟢 BUY++" if decision == "BUY++" else
        "🟩 BUY" if decision == "BUY" else
        "🔴 SELL++" if decision == "SELL++" else
        "🟥 SELL" if decision == "SELL" else
        "⚪ WAIT"
    )

    return {
        "symbol": key,
        "decision": decision,
        "signal_state": signal_state,
        "score": score,
        "confidence": confidence,
        "price": p,
        "rsi": rsi,
        "macd": macd,
        "adx": adx,
        "demarker": dem,
        "volume_ratio": vol_ratio,
        "regime": regime,
        "candle": candle,
        "mtf": mtf,
        "method1": regime_score,
        "method2": liquidity_score,
        "method3": curvature_score,
        "orderbook_score": ob_score,
        "order_block": ob_text,
        "order_block_zone": ob_zone,
        "supports": supports[-6:],
        "resistances": resistances[-6:],
        "nearest_support": support,
        "nearest_resistance": resistance,
        "primary_confirmation": primary,
        "secondary_confirmation": secondary,
        "entry": p if decision != "WAIT" else None,
        "stop_loss": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_reward": rr,
        "reasons": pa_reasons,
        "candles": extended_patterns,
        "eqh": eq_levels.get("eqh", []),
        "eql": eq_levels.get("eql", []),
        "imbalance_score": imbalance_score,
        "bullish_fvg": bullish_fvg,
        "bearish_fvg": bearish_fvg,
        "choch_bullish": bool(bos.get("choch_bullish")),
        "choch_bearish": bool(bos.get("choch_bearish")),
        "bos_bullish": bool(bos.get("bos_bullish")),
        "bos_bearish": bool(bos.get("bos_bearish")),
        "trade_plan": plan,
        "indicators": indx,
        "bb_squeeze": bool(indx.get("bb_squeeze",False)) if indx else False,
        "stoch_k": float(indx.get("stoch_k",50)),
        "stoch_d": float(indx.get("stoch_d",50)),
        "willr": float(indx.get("willr",-50)),
        "cci": float(indx.get("cci",0)),
        "mfi": float(indx.get("mfi",50)),
        "obv_trend": "صعودی" if indx and indx.get("obv",0)>indx.get("obv_ema",0) else "نزولی",
        "vwap": indx.get("vwap"),
        "ema9": float(indx.get("ema9",0)), "ema21": float(indx.get("ema21",0)), "ema50": float(indx.get("ema50",0)),
        "indicator_confluence": float(indx.get("confluence_score",0)) if indx else 0.0,
    }


# ---------------- UI helpers ----------------

def _make_scrollable_tab(notebook, bg="#0f172a"):
    """Create a full-page web-style scrollable tab with BOTH axes.

    The inner page keeps its natural requested width, so wide content can be
    reached with the horizontal scrollbar instead of being clipped. Existing
    Treeview/Text widgets may still have their own local scrollbars.
    """
    outer = tk.Frame(notebook, bg=bg, bd=0, highlightthickness=0)
    canvas = tk.Canvas(
        outer, bg=bg, highlightthickness=0, bd=0,
        xscrollincrement=20, yscrollincrement=20
    )
    vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    hbar = ttk.Scrollbar(outer, orient="horizontal", command=canvas.xview)
    inner = tk.Frame(canvas, bg=bg, bd=0, highlightthickness=0)

    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

    def _sync_region(_event=None):
        try:
            canvas.configure(scrollregion=canvas.bbox("all"))
        except Exception:
            pass

    def _sync_size(event=None):
        try:
            viewport_w = max(1, canvas.winfo_width())
            requested_w = max(1, inner.winfo_reqwidth())
            # Never shrink a wide page below its natural width; otherwise
            # horizontal scrolling would be impossible.
            canvas.itemconfigure(window_id, width=max(viewport_w, requested_w))
            canvas.configure(scrollregion=canvas.bbox("all"))
        except Exception:
            pass

    inner.bind("<Configure>", _sync_region, add="+")
    canvas.bind("<Configure>", _sync_size, add="+")

    canvas.grid(row=0, column=0, sticky="nsew")
    vbar.grid(row=0, column=1, sticky="ns")
    hbar.grid(row=1, column=0, sticky="ew")
    outer.grid_rowconfigure(0, weight=1)
    outer.grid_columnconfigure(0, weight=1)

    inner._scroll_outer = outer
    inner._scroll_canvas = canvas
    inner._scrollbar = vbar
    inner._hscrollbar = hbar
    inner._scroll_window_id = window_id
    outer._scroll_inner = inner
    outer._scroll_canvas = canvas
    outer._scroll_vbar = vbar
    outer._scroll_hbar = hbar
    return inner


def _install_page_scroll_support(root):
    """Route mouse-wheel events to the page-scroll canvas under the pointer.

    Widgets such as Treeview/Text/Listbox already have their own yview and
    `_install_scroll_support`; those bindings run first and keep their local
    scrolling. For ordinary labels/frames/buttons, the whole tab page scrolls.
    """
    if getattr(root, "_page_scroll_support_installed", False):
        return
    root._page_scroll_support_installed = True

    def _find_page_canvas(widget):
        try:
            w = widget
            while w is not None:
                canvas = getattr(w, "_scroll_canvas", None)
                if canvas is not None and canvas.winfo_exists():
                    return canvas
                parent_name = w.winfo_parent()
                if not parent_name:
                    break
                w = w.nametowidget(parent_name)
        except Exception:
            pass
        return None

    def _wheel(event):
        try:
            # If the pointer is over a widget that already has an explicit
            # scrolling binding, don't interfere with it.
            w = event.widget
            cls = str(w.winfo_class())
            if cls in ("Treeview", "Text", "Listbox", "Canvas"):
                # Canvas here may be the page canvas itself; otherwise its
                # own binding handles it.
                if getattr(w, "_scroll_canvas", None) is not None:
                    pass
                else:
                    return None

            canvas = _find_page_canvas(w)
            if canvas is None:
                return None

            delta = getattr(event, "delta", 0)
            if delta:
                units = -int(delta / 120)
                if units == 0:
                    units = -1 if delta > 0 else 1
            else:
                # Windows fallback; Linux/macOS may provide num instead.
                units = -1 if getattr(event, "num", None) == 4 else 1

            canvas.yview_scroll(units, "units")
            return "break"
        except Exception:
            return None

    def _shift_wheel(event):
        try:
            canvas = _find_page_canvas(event.widget)
            if canvas is None:
                return None
            delta = getattr(event, "delta", 0)
            units = -int(delta / 120) if delta else 0
            if units == 0:
                units = -1 if delta > 0 else 1
            canvas.xview_scroll(units, "units")
            return "break"
        except Exception:
            return None

    # bind_all is intentionally used only for page-level routing. Existing
    # widget bindings with "break" remain authoritative for tables/text.
    root.bind_all("<MouseWheel>", _wheel, add="+")
    root.bind_all("<Shift-MouseWheel>", _shift_wheel, add="+")

def _install_scroll_support(widget, canvas=None):
    """Add mouse-wheel scrolling to a Tk widget without breaking horizontal scroll."""
    try:
        def on_wheel(event):
            try:
                delta = int(-1 * (event.delta / 120))
            except Exception:
                delta = -1 if getattr(event, "delta", 0) < 0 else 1
            try:
                widget.yview_scroll(delta, "units")
                return "break"
            except Exception:
                return None

        widget.bind("<MouseWheel>", on_wheel, add="+")
        # Shift+wheel for horizontal scrolling.
        def on_shift_wheel(event):
            try:
                delta = int(-1 * (event.delta / 120))
            except Exception:
                delta = -1 if getattr(event, "delta", 0) < 0 else 1
            try:
                widget.xview_scroll(delta, "units")
                return "break"
            except Exception:
                return None

        widget.bind("<Shift-MouseWheel>", on_shift_wheel, add="+")
        widget.bind("<Up>", lambda e: (widget.yview_scroll(-1, "units"), "break")[1], add="+")
        widget.bind("<Down>", lambda e: (widget.yview_scroll(1, "units"), "break")[1], add="+")
        widget.bind("<Left>", lambda e: (widget.xview_scroll(-1, "units"), "break")[1], add="+")
        widget.bind("<Right>", lambda e: (widget.xview_scroll(1, "units"), "break")[1], add="+")
        widget.bind("<Prior>", lambda e: (widget.yview_scroll(-8, "units"), "break")[1], add="+")
        widget.bind("<Next>", lambda e: (widget.yview_scroll(8, "units"), "break")[1], add="+")
    except Exception:
        pass


def _tree_with_scrollbars(parent, columns, **tree_kwargs):
    """Create a Treeview with both vertical and horizontal scrollbars."""
    frame = tk.Frame(parent)
    frame.pack(fill="both", expand=True)
    tree = ttk.Treeview(frame, columns=columns, show="headings", **tree_kwargs)
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    _install_scroll_support(tree)
    return frame, tree, vsb, hsb


LIVE_TRADING_ENABLED = False
PAPER_TRADING_ENABLED = True
# ---------------- CONFIG / CONSTANTS ----------------
# ===== V8 Persistent AI Cache =====
# Keep learned market data outside VS Code/Program Files folders.
APP_DATA_DIR = Path.home() / "Nobitex_AI_Trader_Pro_Data"
CACHE_DATA_DIR = APP_DATA_DIR / "cache"
ML_DATA_DIR = APP_DATA_DIR / "ml"
BACKTEST_DATA_DIR = APP_DATA_DIR / "backtests"
SIGNAL_SCREENSHOT_DIR = APP_DATA_DIR / "signal_screenshots"
for _d in (APP_DATA_DIR, CACHE_DATA_DIR, ML_DATA_DIR, BACKTEST_DATA_DIR, SIGNAL_SCREENSHOT_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

API_BASE = "https://apiv2.nobitex.ir"
NOBITEX_API_BASES = ("https://apiv2.nobitex.ir", "https://api.nobitex.ir")
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Referer": "https://nobitex.ir"}
REQUEST_TIMEOUT = 7
LOG_TRUNCATE = 800
PRIMARY_RESOLUTION = 15
CONFIRM_RESOLUTION = 60
NUM_CANDLES = 200
INTERVAL = 30
CACHE_TTL = 60
executor_workers = 16

# ===== V8 Reliable Data Engine =====
# Nobitex documents a public UDF history limit of about 60 requests/minute.
# Keep a single gate for history calls so worker threads cannot burst the API.
UDF_MIN_INTERVAL = 0.10  # FAST overview: ~10 requests/sec gate; avoids multi-minute first pass
ORDERBOOK_MIN_INTERVAL = 0.10
UDF_TIMEOUT = (1.5, 3.0)
ORDERBOOK_TIMEOUT = (1.0, 2.5)
HISTORY_NEGATIVE_TTL = 300.0
HISTORY_INVALID_TTL = 6 * 3600.0
HISTORY_TRANSIENT_TTL = 30.0
MAX_HISTORY_CANDIDATES = 2

# ===== Fast Overview mode =====
# The overview table must not perform full multi-timeframe/ML analysis for
# every exchange symbol. One 15m history request is enough to populate the
# visible indicator columns; expensive confirmation/ML/4h work is reserved
# for later candidate analysis. This keeps all 531 symbols visible without
# turning startup into a 20-30 minute serial history sweep.
OVERVIEW_PRIMARY_ONLY = True
OVERVIEW_ENABLE_ML = False
OVERVIEW_ENABLE_BTC_240 = False
OVERVIEW_BATCH_SIZE = 64

class _ApiRateGate:
    def __init__(self, min_interval: float):
        self.min_interval = float(min_interval)
        self.lock = threading.Lock()
        self.last = 0.0
    def wait(self):
        with self.lock:
            now = time.monotonic()
            delay = self.min_interval - (now - self.last)
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()

_udf_gate = _ApiRateGate(UDF_MIN_INTERVAL)
_orderbook_gate = _ApiRateGate(ORDERBOOK_MIN_INTERVAL)

# Per-symbol health prevents endless retries for invalid/empty markets.
_symbol_health_lock = threading.RLock()
_symbol_health = {}

def _history_health(key_sym: str, res: int):
    key = f"{key_sym}:{int(res)}"
    with _symbol_health_lock:
        return dict(_symbol_health.get(key, {}))

def _mark_history_failure(key_sym: str, res: int, reason: str):
    now = time.time()
    transient = any(x in reason for x in ("timeout", "RemoteDisconnected", "ConnectionError", "SSLError", "request-", "http-429", "http-5"))
    invalid = reason.startswith("http-400") or reason in {"no_data", "no-candles", "incomplete-ohlcv"}
    ttl = HISTORY_INVALID_TTL if invalid else (HISTORY_TRANSIENT_TTL if transient else HISTORY_NEGATIVE_TTL)
    health_key = f"{key_sym}:{int(res)}"
    with _symbol_health_lock:
        h = _symbol_health.setdefault(health_key, {"failures":0, "last_reason":None, "last_failure":0.0, "blocked_until":0.0, "ok":False})
        h["failures"] = int(h.get("failures", 0)) + 1
        h["last_reason"] = reason
        h["last_failure"] = now
        h["blocked_until"] = now + ttl
        h["ok"] = False
        return dict(h)

def _mark_history_success(key_sym: str, res: int, bars: int):
    health_key = f"{key_sym}:{int(res)}"
    with _symbol_health_lock:
        _symbol_health[health_key] = {"failures":0, "last_reason":f"ok-{bars}", "last_failure":0.0, "blocked_until":0.0, "ok":True, "last_success":time.time(), "bars":int(bars)}


# Talaye settings
TALAYE_ENABLED = True
TALAYE_MIN_SCORE = 0.78
TALAYE_MIN_SCORE_V2 = 0.85
TALAYE_VOL_FACTOR = 2.0
TALAYE_RSI_RISE_BARS = 3
TALAYE_WEIGHT = 1.5

# Volume & Signal thresholds
MIN_VOL_USDT = 50_000
MIN_VOL_USDT_V2 = 25_000
MIN_RULES_FOR_SIGNAL = 2
MIN_RULES_FOR_SIGNAL_V2 = 1
SIGNAL_COOLDOWN = 15 * 60
MIN_LOG_SCORE = 5
FAILURE_THRESHOLD = 4
EVAL_SECONDS = 3600
EXPIRE_SECONDS = 6 * 3600
MAX_SIGNAL_LOG = 5000
SETTINGS_FILE = "scalper_settings.json"
LEVERAGE = 20
AUTO_TRADE_USDT_VALUE = 5.0  # legacy compatibility; live sizing uses configurable values below
MARGIN_WALLET_USE_PCT = 75.0
MARGIN_TRADE_USDT = 5.0
SPOT_WALLET_USE_PCT = 75.0
SPOT_TRADE_USDT = 5.0
# Fixed RLS notional for IRT markets (3,000,000 Toman = 30,000,000 RLS).
MARGIN_TRADE_RLS = 30_000_000.0
SPOT_TRADE_RLS = 30_000_000.0
# Real trading is deliberately OFF by default. When enabled, Auto Trade submits
# real MARGIN orders through Nobitex /margin/orders/add.
REAL_TRADING_ENABLED = False
REAL_TRADING_CONFIRMATION = ""
REAL_TRADING_CONFIRM_PHRASE = "REAL"
REAL_TRADING_SPOT_ONLY = False
REAL_TRADING_MARGIN_ONLY = True
TRAILING_PERCENT_DEFAULT = 50.0  # percent of profit allowed to retrace after stage 2
TRAILING_ACTIVATE_PCT_DEFAULT = 1.0
TRAILING_STAGE2_PCT_DEFAULT = 2.0
TRAILING_STAGE1_RETRACE_DEFAULT = 70.0
TRAILING_STAGE2_RETRACE_DEFAULT = 50.0
STOP_LOSS_PCT_DEFAULT = 0.3

# Confidence / persistence
MIN_SIGNAL_PERSIST_CANDLES = 2
CONFIDENCE_DECAY_PER_MIN = 0.02

# Telegram thresholds
TELEGRAM_MIN_PROB = 0.60
TELEGRAM_MIN_PRED_MOVE = 0.5
TELEGRAM_COOLDOWN = 15 * 60
TELEGRAM_MIN_PROB_STRICT = 0.65
TELEGRAM_MIN_PRED_MOVE_STRICT = 1.0
TELEGRAM_MIN_RULES = 2
TELEGRAM_PERSIST_BARS = 2
TELEGRAM_CHECK_SHORTER_TF = 15

# Advanced ML Config
ML_ARTIFACT_DIR = "ml_artifacts_enhanced"
ML_SEQUENCE_LENGTH = 48  # LSTM lookback (48 candles = 12h on 15m)
ML_LSTM_UNITS = 64
ML_DROPOUT = 0.3
ML_EPOCHS = 100
ML_BATCH_SIZE = 32
ML_PATIENCE = 10
ML_RETRAIN_INTERVAL = 24 * 3600  # Retrain every 24h

# Risk Management Config
KELLY_FRACTION = 0.3  # Half-Kelly for safety
MAX_DRAWDOWN_PCT = 15.0  # Stop trading if drawdown exceeds 15%
RISK_PER_TRADE_PCT = 1.0  # Default 1% risk per trade
MONTE_CARLO_SIMS = 5000

# Multi-Timeframe Config
MTF_TIMEFRAMES = [1, 5, 15, 60, 240, 1440]  # minutes
MTF_WEIGHTS = {1: 0.05, 5: 0.10, 15: 0.20, 60: 0.30, 240: 0.20, 1440: 0.15}

# Market Structure Config
SWING_LOOKBACK = 5
OB_BARS = 5
FVG_MIN_GAP_PCT = 0.05  # 0.05%
LIQUIDITY_SWEEP_PCT = 0.15  # 0.15%

# Microstructure
VPIN_BARS = 50
ORDERBOOK_DEPTH_LEVELS = 20

USER_SYMBOLS = [
    "BTCIRT","ETHIRT","SOLIRT","TONIRT","DOGEIRT","TRXIRT","XRPIRT","ADAIRT","BNBIRT","LTCIRT",
    "SHIBIRT","PEPEIRT","BONKIRT","AVAXIRT","MATICIRT","DOTIRT","LINKIRT","UNIIRT","BCHIRT","ETCIRT",
    "XLMIRT","AAVEIRT","FILIRT","ATOMIRT","VETIRT","ALGOIRT","EOSIRT","MANAIRT","SANDIRT","GRTIRT",
    "NEARIRT","CHZIRT","AXSIRT","THETAIRT","HBARIRT","FTMIRT","RUNEIRT","APEIRT","GALAIRT","ZECIRT",
    "DASHIRT","XTZIRT","ENJIRT","BATIRT","COMPIRT","MKRIRT","SNXIRT","SUSHIIRT","1INCHIRT","CRVIRT",
    "BALIRT","KNCIRT","STORJIRT","OCEANIRT","RLCIRT","SKLIRT","BNTIRT","RENIRT","LRCIRT","OMGIRT",
    "ZRXIRT","ANTIRT","BANDIRT","NMRIRT","CVCIRT","REPIRT","COTIIRT","FETIRT","ONEIRT","REEFIRT",
    "CELOIRT","KSMIRT","ICPIRT","ARIRT","ROSEIRT","MOVRIRT","GLMIRT","AUDIOIRT","CTSIIRT","MASKIRT",
    "ALICEIRT","LINAIRT","DENTIRT","HOTIRT","SLPIRT","TLMIRT","FLOKIIRT","MEWIRT","NOTIRT","TURBOIRT",
    "BRETTIRT","DEGENIRT","MOGIRT","POPCATIRT","BOMEIRT","WLDIRT","JUPIRT","TNSRIRT","IOIRT","ZEREBROIRT",
    "HNTIRT","PYTHIRT","ONDOIRT","STRKIRT","ENAIRT","WIRT","ZETAIRT","PENDLEIRT","BBIRT","AEVOIRT",
    "PORTALIRT","SAGAIRT","TAIKOIRT","ALTIRT","USDTIRT",
    "BTCUSDT","ETHUSDT","SOLUSDT","TONUSDT","DOGEUSDT","TRXUSDT","XRPUSDT","ADAUSDT","BNBUSDT","LTCUSDT",
    "SHIBUSDT","PEPEUSDT","BONKUSDT","AVAXUSDT","MATICUSDT","DOTUSDT","LINKUSDT","UNIUSDT","BCHUSDT","ETCUSDT",
    "XLMUSDT","AAVEUSDT","FILUSDT","ATOMUSDT","VETUSDT","ALGOUSDT","EOSUSDT","MANAUSDT","SANDUSDT","GRTUSDT",
    "NEARUSDT","CHZUSDT","AXSUSDT","THETAUSDT","HBARUSDT","FTMUSDT","RUNEUSDT","APEUSDT","GALAUSDT","ZECUSDT",
    "DASHUSDT","XTZUSDT","ENJUSDT","BATUSDT","COMPUSDT","MKRUSDT","SNXUSDT","SUSHIUSDT","1INCHUSDT","CRVUSDT",
    "BALUSDT","KNCUSDT","STORJUSDT","OCEANUSDT","RLCUSDT","SKLUSDT","BNTUSDT","RENUSDT","LRCUSDT","OMGUSDT",
    "ZRXUSDT","ANTUSDT","BANDUSDT","NMRUSDT","CVCUSDT","REEFUSDT","COTIUSDT","FETUSDT","ONEUSDT","REEFUSDT",
    "CELOUSDT","KSMUSDT","ICPUSDT","ARUSDT","ROSEUSDT","MOVRUSDT","GLMUSDT","AUDIOUSDT","CTSIUSDT","MASKUSDT",
    "ALICEUSDT","LINAUSDT","DENTUSDT","HOTUSDT","SLPUSDT","TLMUSDT","FLOKIUSDT","MEWUSDT","NOTUSDT","TURBOUSDT",
    "BRETTUSDT","DEGENUSDT","MOGUSDT","POPCATUSDT","BOMEUSDT","WLDUSDT","JUPUSDT","TNSRUSDT","IOUSDT","ZEREBROUSDT",
    "HNTUSDT","PYTHUSDT","ONDOUSDT","STRKUSDT","ENAUSDT","WUSDT","ZETAUSDT","PENDLEUSDT","BBUSDT","AEVOUSDT",
    "PORTALUSDT","SAGAUSDT","TAIKOUSDT","ALTUSDT"
]
SAFE_FALLBACK_SYMBOLS = ["BTCUSDT","ETHUSDT","XRPUSDT","DOGEUSDT","ADAUSDT"]

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = ""
_TELEGRAM_CONFIG_FILE = "telegram_config.json"
_signal_telegram_sent: Dict[str, Dict[str, Any]] = {}

# Logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TRADE_HISTORY_FILE = str(BASE_DIR / "trade_history.csv")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGFILE = str(LOG_DIR / "run_strategy_Version2_Enhanced.log")
logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(LOGFILE, encoding="utf-8"), logging.StreamHandler()])

# ===== Persistent cache helpers =====
def save_json_cache(name, data):
    try:
        import json
        f=CACHE_DATA_DIR/(str(name)+".json")
        f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logging.debug("cache save failed %s", e)

def load_json_cache(name, default=None):
    try:
        import json
        f=CACHE_DATA_DIR/(str(name)+".json")
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        logging.debug("cache load failed %s", e)
    return default

# Globals
cache: Dict[str, Any] = {}
last_delta_cache: Dict[str, float] = {}
_ob_price_cache: Dict[str, Tuple[float, float]] = {}
OB_PRICE_CACHE_TTL = 5.0
signal_log: List[Dict[str, Any]] = []
signal_log_v2: List[Dict[str, Any]] = []

# UI activity monitor: each tab gets its own status/color so long-running
# background jobs are visible without blocking the GUI.
TAB_ACTIVITY_COLORS = {
    "نمای کلی": "#38bdf8",
    "مرکز معاملات هوشمند": "#22c55e",
    "سیگنال‌ها": "#22c55e",
    "اتو ترید (شبیه‌سازی)": "#a78bfa",
    "آزمایشگاه تحلیل عمیق": "#facc15",
    "🤖 AI Trader": "#60a5fa",
    "🧠 اجرای هوشمند": "#f97316",
    "ساختار بازار": "#14b8a6",
    "تحقیق هوشمند / بهینه‌سازی": "#e879f9",
    "Evidence / ارزیابی سیگنال": "#fb7185",
}
TAB_ACTIVITY = {}
TAB_ACTIVITY_LOCK = threading.RLock()
signal_id_counter = 0
signal_id_counter_v2 = 0
auto_trade_log: List[Dict[str, Any]] = []
trade_history: List[Dict[str, Any]] = []
state_lock = threading.RLock()
app = None
valid_symbols_map: Dict[str, Any] = {}
canonical_to_api: Dict[str, str] = {}
SYMBOLS: List[str] = SAFE_FALLBACK_SYMBOLS.copy()
INDICATOR_WEIGHTS = {"rsi":0.3,"macd":0.3,"demarker":0.2,"bollinger":0.1,"vwap":0.1}
talaye_alerts: Dict[str, float] = {}
last_signal_time_per_symbol: Dict[str, float] = {}
last_signal_signature: Dict[str, Any] = {}
pruned_symbols: set = set()
failure_counts: defaultdict = defaultdict(int)
consecutive_loss_count = {"long": 0, "short": 0}
suppress_direction_until = {"long": 0, "short": 0}
SLIPPAGE_LOG = "slippage_log.jsonl"

# Historical backtest execution-cost assumptions. All are configurable via
# environment variables so the research engine never silently assumes zero costs.
# FEE is charged per side; SPREAD is the full bid/ask width; SLIPPAGE is per side.
BACKTEST_FEE_PCT_PER_SIDE = float(os.getenv("V8_BACKTEST_FEE_PCT_PER_SIDE", "0.10"))
BACKTEST_SPREAD_PCT = float(os.getenv("V8_BACKTEST_SPREAD_PCT", "0.05"))
BACKTEST_SLIPPAGE_PCT_PER_SIDE = float(os.getenv("V8_BACKTEST_SLIPPAGE_PCT_PER_SIDE", "0.05"))
BACKTEST_USE_COSTS = os.getenv("V8_BACKTEST_USE_COSTS", "1").strip().lower() not in {"0", "false", "no", "off"}

def backtest_round_trip_cost_pct() -> float:
    """Approximate round-trip execution drag in percentage points."""
    if not BACKTEST_USE_COSTS:
        return 0.0
    return max(0.0, 2.0 * BACKTEST_FEE_PCT_PER_SIDE + BACKTEST_SPREAD_PCT +
               2.0 * BACKTEST_SLIPPAGE_PCT_PER_SIDE)

# ML artifacts cache
_ml_models_loaded = False
_ml_ensemble = None  # Dict of models
_ml_scaler = None
_ml_feature_importance = None
_ml_last_train_time = 0

# Backtest & Risk globals
_backtest_results: List[Dict] = []
_equity_curve: List[float] = [1000.0]  # Start with 1000 USDT
_max_equity = 1000.0
_current_drawdown = 0.0

# Market structure cache
_market_structure_cache: Dict[str, Dict] = {}
_mtf_alignment_cache: Dict[str, float] = {}

# requests session
_session: Optional[requests.Session] = None
def get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        retries = Retry(connect=2, read=0, redirect=1, status=2, backoff_factor=0.6, status_forcelist=(429,500,502,503,504), allowed_methods=frozenset(["GET","POST"]), respect_retry_after_header=True)
        adapter = HTTPAdapter(max_retries=retries, pool_connections=32, pool_maxsize=32)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _session = s
    return _session

# Button style
CTRL_BTN_BG = "#0ea5e9"
CTRL_BTN_FG = "white"
CTRL_BTN_WIDTH = 12
CTRL_BTN_FONT = ("Tahoma", 9, "bold")

# ---------------- Helpers ----------------
def safe_float(x, default=0.0):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return default

def format_price(p):
    try:
        p = float(p)
        if p >= 1000:
            return f"{p:,.0f}"
        if p >= 1:
            return f"{p:,.2f}"
        if p >= 0.01:
            return f"{p:,.4f}"
        return f"{p:,.6f}"
    except Exception:
        return "—"

def normalize_symbol(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()

def canonical(sym: str) -> str:
    return normalize_symbol(sym)


def api_key_to_canonical(api_key: str) -> str:
    """
    Converts Nobitex API keys like 'btc-rls' or 'btc-usdt' to canonical form 'BTCIRT' or 'BTCUSDT'.
    """
    try:
        key = str(api_key).lower().strip()
        key = key.replace("-", "")
        # rls → irt (Toman)
        key = key.replace("rls", "irt")
        return normalize_symbol(key)
    except Exception:
        return normalize_symbol(api_key)

def _save_telegram_config(chat_id: str):
    try:
        cfg = {"chat_id": str(chat_id)}
        with open(_TELEGRAM_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)
    except Exception:
        logging.debug("failed to save telegram config")

def _load_telegram_config():
    try:
        if os.path.exists(_TELEGRAM_CONFIG_FILE):
            with open(_TELEGRAM_CONFIG_FILE, "r", encoding="utf-8") as f:
                j = json.load(f)
                return j.get("chat_id", "")
    except Exception:
        pass
    return ""

# ---------------- Telegram (enhanced) ----------------
def send_telegram(msg: str, debug: bool = False):
    try:
        token = globals().get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat = globals().get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID", "")
        if not chat:
            loaded = _load_telegram_config()
            if loaded:
                chat = loaded
                globals()["TELEGRAM_CHAT_ID"] = chat
        if not token:
            logging.warning("Telegram token not set; skipping send")
            return
        if not chat:
            try:
                url_get = f"https://api.telegram.org/bot{token}/getUpdates"
                r = get_session().get(url_get, timeout=8)
                if r.status_code == 200:
                    j = r.json()
                    results = j.get("result", []) if isinstance(j, dict) else []
                    found = None
                    for item in reversed(results):
                        msg_obj = item.get("message") or item.get("edited_message") or item.get("channel_post") or item.get("edited_channel_post")
                        if not msg_obj and "callback_query" in item and item["callback_query"].get("message"):
                            msg_obj = item["callback_query"]["message"]
                        if msg_obj and isinstance(msg_obj, dict):
                            ch = msg_obj.get("chat", {})
                            chat_id = ch.get("id")
                            if chat_id:
                                found = str(chat_id)
                                break
                    if found:
                        chat = found
                        globals()["TELEGRAM_CHAT_ID"] = chat
                        _save_telegram_config(chat)
                        logging.info("Telegram chat_id auto-discovered: %s", chat)
                    else:
                        logging.warning("Telegram getUpdates: no chat found. Send /start to bot.")
                else:
                    logging.warning("Telegram getUpdates non-200: %s", r.status_code)
            except Exception:
                logging.debug("telegram getUpdates error: %s", traceback.format_exc())
        if not chat:
            logging.warning("Telegram chat_id not set; skipping send.")
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        proxies = None
        for k in ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"]:
            v = os.getenv(k)
            if v:
                proxies = proxies or {}
                proxies["https" if "HTTPS" in k.upper() else "http"] = v
        s = get_session()
        try:
            r2 = s.post(url, data={"chat_id": chat, "text": msg, "parse_mode": "HTML"}, timeout=10, proxies=proxies)
            if debug:
                logging.info("telegram send status=%s", getattr(r2, "status_code", None))
            if getattr(r2, "status_code", None) != 200:
                logging.warning("Telegram API non-200: %s", getattr(r2, "status_code", None))
        except Exception as e:
            logging.exception("send_telegram network error: %s", e)
    except Exception:
        logging.debug("send_telegram outer error: %s", traceback.format_exc())

def _make_telegram_signature_for_rec(rec: dict) -> str:
    try:
        s = f"{rec.get('symbol')}_{rec.get('side')}_{rec.get('predicted_move_pct')}_{round(rec.get('prob_up', rec.get('probability',0)), 2)}"
        return s
    except Exception:
        return str(time.time())

def should_send_telegram_for_signal(rec: dict) -> bool:
    try:
        prob = rec.get("prob_up", None)
        if prob is None:
            prob = rec.get("probability", 0.0) / 100.0 if rec.get("probability") is not None else 0.0
        if prob > 1.1:
            prob = prob / 100.0
        pred_move = rec.get("predicted_move_pct", None) or 0.0
        if prob < TELEGRAM_MIN_PROB or pred_move < TELEGRAM_MIN_PRED_MOVE:
            return False
        sym = canonical(rec.get("symbol", ""))
        sig = _make_telegram_signature_for_rec(rec)
        now_ts = time.time()
        prev = _signal_telegram_sent.get(sym)
        if prev:
            if prev.get("sig") == sig and (now_ts - prev.get("ts", 0)) < TELEGRAM_COOLDOWN:
                return False
        return True
    except Exception:
        return False

def _count_recent_signal_occurrences(sym: str, side: str, lookback_seconds: int = TELEGRAM_PERSIST_BARS * 3600) -> int:
    try:
        key = canonical(sym)
        cutoff = time.time() - lookback_seconds
        cnt = 0
        with state_lock:
            for r in signal_log:
                if r.get("symbol") == key and r.get("side") == side and (not r.get("result")) and (not r.get("expired")):
                    if r.get("time", 0) >= cutoff:
                        cnt += 1
        return cnt
    except Exception:
        return 0

def _confirm_on_shorter_tf(sym: str, side: str, tf_minutes: int = TELEGRAM_CHECK_SHORTER_TF) -> bool:
    try:
        df = get_candles_cached(sym, tf_minutes, n=6, valid_symbols_map=valid_symbols_map)
        if df is None or df.empty or len(df) < 3:
            return False
        last3 = df["close"].iloc[-3:].mean()
        prev3 = df["close"].iloc[-6:-3].mean()
        if side == "لانگ":
            return last3 > prev3
        elif side == "شورت":
            return last3 < prev3
        return False
    except Exception:
        return False

def is_signal_special(rec: dict) -> bool:
    try:
        prob = rec.get("prob_up", None)
        if prob is None:
            prob = rec.get("probability", 0.0) / 100.0 if rec.get("probability") is not None else 0.0
        if prob > 1.1:
            prob = prob / 100.0
        pred = rec.get("predicted_move_pct", 0.0)
        if prob < TELEGRAM_MIN_PROB_STRICT or pred < TELEGRAM_MIN_PRED_MOVE_STRICT:
            return False
        if rec.get("talaye"):
            return True
        if len(rec.get("matched_rules", []) or []) >= TELEGRAM_MIN_RULES:
            return True
        if _confirm_on_shorter_tf(rec.get("symbol",""), rec.get("side","")):
            return True
        occurrences = _count_recent_signal_occurrences(rec.get("symbol",""), rec.get("side",""))
        if occurrences >= TELEGRAM_PERSIST_BARS:
            return True
        # NEW: Check MTF alignment
        mtf = _mtf_alignment_cache.get(canonical(rec.get("symbol","")), 0.0)
        if mtf >= 0.7 and prob >= 0.60:
            return True
        return False
    except Exception:
        return False

def _trim_signal_log():
    try:
        with state_lock:
            while len(signal_log) > MAX_SIGNAL_LOG:
                signal_log.pop(0)
            while len(signal_log_v2) > MAX_SIGNAL_LOG:
                signal_log_v2.pop(0)
    except Exception:
        pass

def send_signal_to_telegram(rec: dict):
    try:
        if not is_signal_special(rec):
            return
        if not should_send_telegram_for_signal(rec):
            return
        sym = rec.get("symbol", "")
        side = rec.get("side", "")
        prob = rec.get("prob_up", rec.get("probability", 0.0))
        try:
            prob = float(prob)
            if prob > 1.1:
                prob = prob / 100.0
            prob_pct = prob * 100.0
        except Exception:
            prob_pct = 0.0
        move = rec.get("predicted_move_pct", 0.0)
        target = rec.get("target_price")
        entry = rec.get("entry")
        rules = ", ".join(rec.get("matched_rules_names", []) or [])
        # NEW: Add MTF alignment and confidence
        mtf = _mtf_alignment_cache.get(canonical(sym), 0.0)
        conf = rec.get("confidence", 0.0)
        talaye_txt = " | طلایه سیاه" if rec.get("talaye") else ""
        msg = (
            f"سیگنال ویژه{talaye_txt}\n"
            f"نماد: {sym}\n"
            f"پوزیشن: {side}\n"
            f"ورود: {format_price(entry)}\n"
            f"احتمال: {prob_pct:.1f}%\n"
            f"حرکت پیش‌بینی‌شده: {move}%\n"
            f"قیمت هدف: {format_price(target)}\n"
            f"هم‌راستایی تایم‌فریم‌ها: {mtf:.0%}\n"
            f"اعتبار: {conf:.1f}%\n"
            f"قوانین: {rules or '—'}"
        )
        send_telegram(msg)
        with state_lock:
            _signal_telegram_sent[canonical(sym)] = {"sig": _make_telegram_signature_for_rec(rec), "ts": time.time()}
        logging.info("telegram signal sent for %s (%s)", sym, side)
    except Exception:
        logging.debug("send_signal_to_telegram error: %s", traceback.format_exc())

# ---------------- TTS ----------------
tts_engine = None
tts_queue: Queue = Queue()

def init_tts():
    global tts_engine
    if not TTS_AVAILABLE:
        return
    try:
        tts_engine = pyttsx3.init()
        for v in tts_engine.getProperty("voices"):
            if "fa" in v.id.lower() or "persian" in v.name.lower():
                tts_engine.setProperty("voice", v.id)
                break
        tts_engine.setProperty("rate", 160)
        threading.Thread(target=tts_worker, daemon=True).start()
    except Exception:
        logging.debug("tts init failed")

def tts_worker():
    global tts_engine
    while True:
        try:
            txt = tts_queue.get(timeout=1)
            if txt is None:
                break
            if tts_engine:
                tts_engine.say(txt)
                tts_engine.runAndWait()
            tts_queue.task_done()
        except Exception:
            continue

def say_farsi(txt: str, priority: bool=False):
    if priority:
        try:
            import winsound
            winsound.Beep(1400, 200)
        except Exception:
            pass
    if TTS_AVAILABLE and tts_engine:
        try:
            tts_queue.put_nowait(txt)
        except Exception:
            pass

# ---------------- UI Helpers ----------------
def get_trailing_percent() -> float:
    # Backward-compatible alias: this is the stage-2 retracement percentage.
    try:
        return float(app.trailing_stage2_retrace_var.get())
    except Exception:
        return TRAILING_STAGE2_RETRACE_DEFAULT

def get_trailing_activate() -> float:
    try:
        return float(app.trailing_activate_var.get())
    except Exception:
        return TRAILING_ACTIVATE_PCT_DEFAULT

def get_trailing_stage2() -> float:
    try:
        return float(app.trailing_stage2_var.get())
    except Exception:
        return TRAILING_STAGE2_PCT_DEFAULT

def get_trailing_stage1_retrace() -> float:
    try:
        return float(app.trailing_stage1_retrace_var.get())
    except Exception:
        return TRAILING_STAGE1_RETRACE_DEFAULT

def get_trailing_stage2_retrace() -> float:
    try:
        return float(app.trailing_stage2_retrace_var.get())
    except Exception:
        return TRAILING_STAGE2_RETRACE_DEFAULT

def is_trailing_enabled() -> bool:
    """Return the current global/UI trailing policy safely from any thread."""
    try:
        return bool(app.trailing_enabled.get())
    except Exception:
        # Keep the strategy fail-safe: if the UI is not ready/unavailable,
        # do not silently disable risk management logic.
        return True

def get_stop_loss_pct() -> float:
    try:
        return float(app.stop_loss_pct_var.get())
    except Exception:
        return STOP_LOSS_PCT_DEFAULT

def is_auto_trade_enabled() -> bool:
    try:
        return bool(app.auto_trade.get())
    except Exception:
        return False

def _save_settings():
    try:
        s = {
            "TALAYE_MIN_SCORE": TALAYE_MIN_SCORE,
            "trailing_enabled": bool(app.trailing_enabled.get()) if app else True,
            "trailing_activate_pct": float(app.trailing_activate_var.get()) if app else TRAILING_ACTIVATE_PCT_DEFAULT,
            "trailing_stage2_pct": float(app.trailing_stage2_var.get()) if app else TRAILING_STAGE2_PCT_DEFAULT,
            "trailing_stage1_retrace_pct": float(app.trailing_stage1_retrace_var.get()) if app else TRAILING_STAGE1_RETRACE_DEFAULT,
            "trailing_stage2_retrace_pct": float(app.trailing_stage2_retrace_var.get()) if app else TRAILING_STAGE2_RETRACE_DEFAULT,
            "defensive_shield_pct": float(app.stop_loss_pct_var.get()) if app else STOP_LOSS_PCT_DEFAULT,
            "margin_wallet_use_pct": float(app.margin_wallet_use_pct_var.get()) if app else MARGIN_WALLET_USE_PCT,
            "margin_trade_usdt": float(app.margin_trade_usdt_var.get()) if app else MARGIN_TRADE_USDT,
            "spot_wallet_use_pct": float(app.spot_wallet_use_pct_var.get()) if app else SPOT_WALLET_USE_PCT,
            "spot_trade_usdt": float(app.spot_trade_usdt_var.get()) if app else SPOT_TRADE_USDT,
            "margin_trade_rls": float(app.margin_trade_rls_var.get()) if app else MARGIN_TRADE_RLS,
            "spot_trade_rls": float(app.spot_trade_rls_var.get()) if app else SPOT_TRADE_RLS,
        }
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception:
        logging.debug("failed to save settings")

_RISK_SETTINGS = {}
def _load_settings():
    global TALAYE_MIN_SCORE, _RISK_SETTINGS
    global MARGIN_WALLET_USE_PCT, MARGIN_TRADE_USDT, SPOT_WALLET_USE_PCT, SPOT_TRADE_USDT, MARGIN_TRADE_RLS, SPOT_TRADE_RLS
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
                if "TALAYE_MIN_SCORE" in s:
                    TALAYE_MIN_SCORE = float(s["TALAYE_MIN_SCORE"])
                    logging.info("Loaded TALAYE_MIN_SCORE: %.3f", TALAYE_MIN_SCORE)
                _RISK_SETTINGS = dict(s)
                MARGIN_WALLET_USE_PCT = max(1.0, min(100.0, safe_float(s.get("margin_wallet_use_pct"), MARGIN_WALLET_USE_PCT)))
                MARGIN_TRADE_USDT = max(0.01, safe_float(s.get("margin_trade_usdt"), MARGIN_TRADE_USDT))
                SPOT_WALLET_USE_PCT = max(1.0, min(100.0, safe_float(s.get("spot_wallet_use_pct"), SPOT_WALLET_USE_PCT)))
                SPOT_TRADE_USDT = max(0.01, safe_float(s.get("spot_trade_usdt"), SPOT_TRADE_USDT))
                MARGIN_TRADE_RLS = max(1000.0, safe_float(s.get("margin_trade_rls"), MARGIN_TRADE_RLS))
                SPOT_TRADE_RLS = max(1000.0, safe_float(s.get("spot_trade_rls"), SPOT_TRADE_RLS))
                logging.info("Loaded capital controls: Margin %.1f%% / %.4f USDT, Spot %.1f%% / %.4f USDT",
                             MARGIN_WALLET_USE_PCT, MARGIN_TRADE_USDT, SPOT_WALLET_USE_PCT, SPOT_TRADE_USDT)
    except Exception:
        pass

try:
    _load_settings()
except Exception:
    pass

try:
    _load_pruned_symbols()
except Exception:
    pass

def adjust_talaye(delta: float):
    global TALAYE_MIN_SCORE, TALAYE_MIN_SCORE_V2
    try:
        with state_lock:
            TALAYE_MIN_SCORE = max(0.05, min(1.0, round(TALAYE_MIN_SCORE + delta, 3)))
            TALAYE_MIN_SCORE_V2 = TALAYE_MIN_SCORE
            logging.info("TALAYE_MIN_SCORE adjusted to %.3f", TALAYE_MIN_SCORE)
            try:
                _save_settings()
            except Exception:
                pass
        try:
            if app and hasattr(app, "talaye_user_value"):
                app.talaye_user_value.set(TALAYE_MIN_SCORE)
            if app and hasattr(app, "update_talaye_label"):
                app.root.after(0, app.update_talaye_label)
        except Exception:
            pass
    except Exception as e:
        logging.debug("adjust_talaye error: %s", e)

# ---------------- Confidence & Prediction Helpers ----------------
def compute_signal_confidence(score: float, talaye_score: float, matched_rules_count: int, mtf_align: float = 0.0) -> float:
    try:
        base = abs(score) * 8.0
        base += talaye_score * 100.0
        base += matched_rules_count * 7.0
        base += mtf_align * 20.0  # NEW: MTF alignment bonus
        base = max(30.0, base)
        return min(100.0, round(base, 1))
    except Exception:
        return 30.0

def decay_confidence(initial_conf: float, elapsed_seconds: float) -> float:
    try:
        minutes = max(0.0, elapsed_seconds / 60.0)
        return initial_conf * math.exp(-CONFIDENCE_DECAY_PER_MIN * minutes)
    except Exception:
        return initial_conf

def get_current_confidence(rec: dict) -> float:
    try:
        last = rec.get("last_update", rec.get("created_at", rec.get("time", time.time())))
        elapsed = max(0.0, time.time() - last)
        return round(decay_confidence(rec.get("confidence", 0.0), elapsed), 1)
    except Exception:
        return rec.get("confidence", 0.0)

def estimate_predicted_move(score: float, talaye_score: float, matched_rules_count: int, mtf_align: float = 0.0) -> float:
    try:
        w_score = abs(score) * 0.35
        w_talaye = talaye_score * 8.0
        w_rules = matched_rules_count * 0.9
        w_mtf = mtf_align * 3.0  # NEW
        base = w_score + w_talaye + w_rules + w_mtf
        base = max(0.2, min(20.0, base))
        return round(base, 2)
    except Exception:
        return 1.0

# ---------------- Network & Orderbook (Robust) ----------------
def fetch_market_stats() -> Dict[str, Any]:
    time.sleep(0.15)  # FIX 2: rate limit
    url = urljoin(API_BASE, "/market/stats")
    try:
        r = get_session().get(url, timeout=REQUEST_TIMEOUT)
        logging.info("market/stats status=%s", r.status_code)
        if r.status_code != 200:
            logging.warning("market/stats non-200: %s body=%s", r.status_code, getattr(r, "text", "")[:LOG_TRUNCATE])
            return {}
        j = r.json()
        stats = j.get("stats") or j.get("result") or j.get("data") or j.get("markets") or []
        symbol_map: Dict[str, Any] = {}
        if isinstance(stats, dict):
            for api_key, data in stats.items():
                price = None
                if isinstance(data, dict):
                    price = data.get("latest") or data.get("lastTradePrice") or data.get("price") or data.get("mark")
                # Convert api_key like 'btc-rls' → 'BTCIRT', 'btc-usdt' → 'BTCUSDT'
                can = api_key_to_canonical(api_key)
                symbol_map[can] = price
                # Also keep the raw key for reverse lookup
                canonical_to_api[can] = api_key
        else:
            for item in stats:
                if isinstance(item, dict):
                    api_key = item.get("symbol") or item.get("pair") or item.get("id")
                    price = item.get("price") or item.get("lastTradePrice") or item.get("latest")
                    if api_key:
                        can = api_key_to_canonical(api_key)
                        symbol_map[can] = price
                        canonical_to_api[can] = api_key
        logging.info("Fetched %d symbols from market/stats", len(symbol_map))
        return symbol_map
    except Exception as e:
        logging.error("fetch_market_stats error: %s", e)
        return {}

def _candidate_symbol_forms(sym: str) -> List[str]:
    forms: List[str] = []
    try:
        forms.append(sym)
        forms.append(sym.lower())
        su = str(sym).upper()
        # Generate dash form: BTCIRT → btc-irt, BTCUSDT → btc-usdt
        # FIX: IRT is 3 chars, USDT is 4 chars
        if su.endswith("USDT"):
            base = sym[:-4]
            quote = sym[-4:]
            forms.append(f"{base.lower()}-{quote.lower()}")
        elif su.endswith("IRT"):
            base = sym[:-3]
            quote = sym[-3:]
            forms.append(f"{base.lower()}-{quote.lower()}")
            # Also generate rls form for IRT pairs (Nobitex uses rls internally)
            forms.append(f"{base.lower()}-rls")
        # Reverse: if canonical maps to api key like btc-rls
        api_sym = canonical_to_api.get(normalize_symbol(sym))
        if api_sym:
            forms.extend([api_sym, api_sym.lower(), api_sym.replace("-", ""), api_sym.replace("-", "").lower()])
            if api_sym.lower().endswith("usdt") and "-" not in api_sym:
                base = api_sym[:-4]
                forms.append(f"{base.lower()}-usdt")
            if api_sym.lower().endswith("rls") and "-" in api_sym:
                base = api_sym.split("-")[0]
                forms.append(f"{base}-rls")
                forms.append(f"{base}rls")
    except Exception:
        pass
    seen = set()
    out = []
    for f in forms:
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out

def _save_pruned_symbols():
    try:
        with open("pruned_symbols.txt", "w", encoding="utf-8") as f:
            for s in sorted(pruned_symbols):
                f.write(s + "\n")
    except Exception:
        pass

def _load_pruned_symbols():
    global pruned_symbols
    try:
        if os.path.exists("pruned_symbols.txt"):
            with open("pruned_symbols.txt", "r", encoding="utf-8") as f:
                pruned_symbols = set(line.strip() for line in f if line.strip())
                logging.info("Loaded %d pruned symbols from disk", len(pruned_symbols))
    except Exception:
        pruned_symbols = set()

def robust_orderbook(sym: str, valid_symbols: Optional[set] = None) -> Tuple[bool, Any]:
    """Fast orderbook fetch using the exchange's mapped symbol first."""
    key_sym = canonical(sym)

    if key_sym in pruned_symbols:
        # A prior transient failure should not make a valid market disappear.
        # Remove it when the exchange currently reports the symbol.
        if valid_symbols and key_sym in valid_symbols:
            try:
                pruned_symbols.discard(key_sym)
            except Exception:
                pass
        else:
            return False, {"error": "pruned"}

    if valid_symbols and key_sym not in valid_symbols:
        return False, {"error": "not_in_valid_symbols"}

    # The market/stats response contains the exchange's actual symbol key.
    mapped = canonical_to_api.get(key_sym)
    candidates = [str(mapped)] if mapped else list(_candidate_symbol_forms(sym))
    seen = set()
    candidates = [x for x in candidates if x and not (x in seen or seen.add(x))][:2]

    for cs in candidates:
        # v3 accepts the symbol in the URL path.
        u = f"{API_BASE}/v3/orderbook/{cs}"
        try:
            _orderbook_gate.wait()
            r = get_session().get(u, timeout=ORDERBOOK_TIMEOUT)
            if r.status_code == 400:
                break
            if r.status_code != 200:
                continue
            j = r.json()
            j2 = j
            if isinstance(j, dict):
                if isinstance(j.get("result"), dict):
                    j2 = j["result"]
                elif isinstance(j.get("data"), dict):
                    j2 = j["data"]

            if isinstance(j2, dict):
                bids = j2.get("bids") or j2.get("buy") or j2.get("b") or []
                asks = j2.get("asks") or j2.get("sell") or j2.get("a") or []
                if bids and asks:
                    try:
                        bid_vol = sum(
                            safe_float(x[1]) if isinstance(x, (list, tuple)) and len(x) > 1
                            else safe_float(x.get("quantity") if isinstance(x, dict) else 0)
                            for x in bids[:10]
                        )
                        ask_vol = sum(
                            safe_float(x[1]) if isinstance(x, (list, tuple)) and len(x) > 1
                            else safe_float(x.get("quantity") if isinstance(x, dict) else 0)
                            for x in asks[:10]
                        )
                        total = bid_vol + ask_vol
                        delta = (bid_vol - ask_vol) / total if total > 0 else 0.0
                        with state_lock:
                            last_delta_cache[key_sym] = delta
                            hist = cache.get(f"delta_hist_{key_sym}")
                            if not isinstance(hist, list):
                                hist = []
                            hist.append(float(delta))
                            cache[f"delta_hist_{key_sym}"] = hist[-50:]
                    except Exception:
                        pass
                    with state_lock:
                        failure_counts[key_sym] = 0
                    return True, {"bids": bids, "asks": asks, "raw": j2}

                for fld in ("lastTradePrice", "latest", "price", "last"):
                    if j2.get(fld) is not None:
                        with state_lock:
                            failure_counts[key_sym] = 0
                        return True, j2
        except requests.Timeout:
            continue
        except requests.RequestException:
            continue
        except Exception:
            continue

    # Do not prune on transient network/endpoint failures.
    with state_lock:
        failure_counts[key_sym] += 1
        if failure_counts[key_sym] in (1, 3, 10):
            logging.warning(
                "orderbook unavailable for %s (failures=%d)",
                key_sym,
                failure_counts[key_sym]
            )
    return False, {"error": "unavailable", "symbol": key_sym}


def _pick_orderbook_price(level):
    """Extract a price from a Nobitex orderbook level in list/dict form."""
    try:
        first = level[0] if isinstance(level, (list, tuple)) and level else level
        if isinstance(first, (list, tuple)) and first:
            return safe_float(first[0])
        if isinstance(first, dict):
            return safe_float(first.get("price") or first.get("p") or first.get("rate") or first.get("px"))
        if isinstance(first, (int, float, str)):
            return safe_float(first)
    except Exception:
        pass
    return 0.0


def _extract_quote_from_orderbook_payload(j) -> Dict[str, float]:
    """Return best bid/ask/last/mid from a normalized orderbook payload."""
    out = {"bid": 0.0, "ask": 0.0, "last": 0.0, "mid": 0.0}
    try:
        containers = [j]
        if isinstance(j, dict):
            if isinstance(j.get("raw"), dict):
                containers.insert(0, j["raw"])
            for name in ("result", "data"):
                if isinstance(j.get(name), dict):
                    containers.insert(0, j[name])
        for obj in containers:
            if not isinstance(obj, dict):
                continue
            if not out["last"]:
                for fld in ("lastTradePrice", "latest", "price", "last"):
                    v = safe_float(obj.get(fld))
                    if v > 0:
                        out["last"] = v
                        break
            bids = obj.get("bids") or obj.get("buy") or obj.get("b") or []
            asks = obj.get("asks") or obj.get("sell") or obj.get("a") or []
            if bids and not out["bid"]:
                out["bid"] = _pick_orderbook_price(bids[0])
            if asks and not out["ask"]:
                out["ask"] = _pick_orderbook_price(asks[0])
            if out["bid"] and out["ask"]:
                break
        if out["bid"] > 0 and out["ask"] > 0:
            out["mid"] = (out["bid"] + out["ask"]) / 2.0
        else:
            out["mid"] = out["last"] or out["bid"] or out["ask"]
    except Exception:
        pass
    return out


def _get_live_quote_uncached(sym: str, valid_symbols_map: Optional[Dict[str, Any]]=None) -> Optional[Dict[str, float]]:
    """Fetch a fresh quote. This deliberately bypasses the stale market/stats map."""
    ok, j = robust_orderbook(sym, valid_symbols=set(valid_symbols_map.keys()) if valid_symbols_map else None)
    if not ok:
        return None
    quote = _extract_quote_from_orderbook_payload(j)
    if max(quote.values()) <= 0:
        return None
    return quote


def get_live_trade_quote(symbol: str, side: Optional[str]=None) -> Optional[Dict[str, float]]:
    """Fresh executable quote for a trade. LONG exits use bid; SHORT exits use ask."""
    key_sym = canonical(symbol)
    try:
        q = _get_live_quote_uncached(symbol, valid_symbols_map)
        if q and max(q.values()) > 0:
            with state_lock:
                if q.get("mid", 0) > 0:
                    _ob_price_cache[key_sym] = (q["mid"], time.time())
            return q
    except Exception:
        pass
    # Trading paths must not turn a last-trade/market-stats price into a fake
    # executable bid/ask. If the two-sided quote is unavailable, return None
    # and let the caller WAIT/retry instead of assuming zero spread.
    return None


def get_executable_price(symbol: str, side: Optional[str]=None) -> Optional[float]:
    """Return the side-aware executable price, with sensible fallbacks."""
    q = get_live_trade_quote(symbol, side)
    if not q:
        return None
    side = str(side or "").lower()
    # Trading execution is intentionally strict: LONG must use ASK and SHORT
    # must use BID. No last/mid fallback is allowed for an executable price.
    if side == "long" and q.get("ask", 0) > 0:
        return q["ask"]
    if side == "short" and q.get("bid", 0) > 0:
        return q["bid"]
    return None


def get_price_from_orderbook(sym: str, valid_symbols_map: Optional[Dict[str, Any]]=None) -> Optional[float]:
    """Legacy/general price helper. Uses cached market data for non-trading UI paths."""
    key_sym = canonical(sym)
    if valid_symbols_map:
        p = valid_symbols_map.get(key_sym)
        if p is not None:
            fp = safe_float(p)
            if fp > 0:
                _ob_price_cache[key_sym] = (fp, time.time())
                return fp
    try:
        _e = _ob_price_cache.get(key_sym)
        if _e and (time.time() - _e[1]) < OB_PRICE_CACHE_TTL:
            return _e[0]
    except Exception:
        pass
    if key_sym in pruned_symbols:
        return None
    p = _get_price_from_orderbook_uncached(sym, valid_symbols_map)
    if p is not None and p > 0:
        _ob_price_cache[key_sym] = (p, time.time())
    return p


def _get_price_from_orderbook_uncached(sym: str, valid_symbols_map: Optional[Dict[str, Any]]=None) -> Optional[float]:
    q = _get_live_quote_uncached(sym, valid_symbols_map)
    if not q:
        return None
    return q.get("last") or q.get("mid") or q.get("bid") or q.get("ask")

# ---------------- History / Candles ----------------
def robust_history_udf(
    sym: str,
    resolution_minutes: int,
    from_ts: int,
    to_ts: int,
    valid_symbols: Optional[set] = None
):
    """V8 history gateway: validation, throttling, classification and cooldown."""
    key_sym = canonical(sym)
    res = int(resolution_minutes)
    if valid_symbols and key_sym not in valid_symbols:
        return None

    health = _history_health(key_sym, res)
    if health.get("blocked_until", 0.0) > time.time():
        return None

    mapped = canonical_to_api.get(key_sym)
    candidates = []
    if mapped:
        candidates.append(str(mapped))
    # Generated forms are only used when no exact mapping exists. This avoids
    # repeatedly trying malformed aliases for symbols that already have a mapping.
    if not candidates:
        candidates.extend(_candidate_symbol_forms(sym))
    seen = set()
    candidates = [x for x in candidates if x and not (x in seen or seen.add(x))][:MAX_HISTORY_CANDIDATES]
    if not candidates:
        _mark_history_failure(key_sym, res, "no-symbol-mapping")
        return None

    requested = max(10, min(500, int((max(1, int(to_ts)-int(from_ts)) / max(1, res*60)) + 1)))

    def _request(cs, rres, fts, tts, countback):
        params = {"symbol": cs, "resolution": int(rres), "from": int(fts), "to": int(tts), "countback": int(min(500, max(10, countback)))}
        try:
            _udf_gate.wait()
            r = get_session().get(f"{API_BASE}/market/udf/history", params=params, timeout=UDF_TIMEOUT)
            if r.status_code != 200:
                return None, f"http-{r.status_code}"
            try:
                j = r.json()
            except Exception:
                return None, "invalid-json"
            if not isinstance(j, dict):
                return None, "invalid-json"
            status = str(j.get("s") or "").lower()
            if status in {"no_data", "no data", "nodata"}:
                return None, "no_data"
            c, o, h, l, v = (j.get(k) or [] for k in ("c","o","h","l","v"))
            if not all(isinstance(x, list) for x in (o,h,l,c,v)):
                return None, "incomplete-ohlcv"
            n = min(len(o), len(h), len(l), len(c), len(v))
            if n < 10:
                return None, "no_data"
            return j, f"ok-{n}"
        except requests.Timeout:
            return None, "timeout"
        except requests.RequestException as exc:
            return None, f"request-{type(exc).__name__}"
        except Exception as exc:
            return None, f"parse-{type(exc).__name__}"

    last_reason = "unknown"
    for cs in candidates:
        j, reason = _request(cs, res, from_ts, to_ts, requested)
        last_reason = reason
        if j is not None:
            _mark_history_success(key_sym, res, int(reason.split("-")[-1]) if reason.startswith("ok-") else requested)
            with state_lock:
                failure_counts[key_sym] = 0
            return j
        # 400 means this exact mapped symbol/request is not valid; do not hammer aliases.
        if reason == "http-400":
            break

    # Never substitute another timeframe for the requested timeframe. Returning
    # 60m data from a 15m request silently corrupts indicators and SMC validation.
    h = _mark_history_failure(key_sym, res, last_reason)
    with state_lock:
        failure_counts[key_sym] += 1
        fc = failure_counts[key_sym]
    if fc in (1,3,10) or last_reason == "http-400":
        logging.warning("history unavailable for %s res=%s failures=%d reason=%s cooldown=%ds", key_sym, res, fc, last_reason, max(0, int(h.get("blocked_until",0)-time.time())))
    return None

def get_candles_cached(sym: str, res: int, n: int=NUM_CANDLES, valid_symbols_map: Optional[Dict[str, Any]]=None) -> pd.DataFrame:
    key_sym = canonical(sym)
    key = f"c_{key_sym}_{res}"
    try:
        if key in cache and time.time() - cache[key]["t"] < CACHE_TTL:
            return cache[key]["df"]
    except Exception:
        pass
    end = int(time.time())
    start = end - n * res * 60
    j = robust_history_udf(sym, res, start, end, valid_symbols=set(valid_symbols_map.keys()) if valid_symbols_map else None)
    if not j:
        return pd.DataFrame()
    try:
        o = j.get("o") or j.get("open") or []
        h = j.get("h") or j.get("high") or []
        l = j.get("l") or j.get("low") or []
        c = j.get("c") or j.get("close") or []
        v = j.get("v") or j.get("volume") or []
        t = j.get("t") or j.get("time") or []
        if not (len(c) >= 10 and len(v) >= 10):
            return pd.DataFrame()
        minlen = min(len(o or []), len(h or []), len(l or []), len(c or []), len(v or []))
        if minlen == 0:
            if len(c) >= 10 and len(v) >= 10:
                df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": v}).astype(float)
                df.index = pd.to_datetime(t[:len(c)], unit="s") if t else pd.date_range(end=pd.Timestamp.now(), periods=len(c), freq=f"{res}min")
                cache[key] = {"df": df, "t": time.time()}
                return df
            return pd.DataFrame()
        o = o[-minlen:]
        h = h[-minlen:]
        l = l[-minlen:]
        c = c[-minlen:]
        v = v[-minlen:]
        t = t[-minlen:] if t else None
        df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).astype(float)
        if t:
            try:
                df.index = pd.to_datetime(t, unit="s")
            except Exception:
                df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq=f"{res}min")
        else:
            df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq=f"{res}min")
        cache[key] = {"df": df, "t": time.time()}
        return df
    except Exception as e:
        logging.debug("candles parse error %s: %s", sym, e)
        return pd.DataFrame()

# ---------------- Advanced Indicators ----------------
def compute_rsi_series(df: pd.DataFrame, p: int=14) -> pd.Series:
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)
    close = df["close"]
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1.0/p, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0/p, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_rsi(df: pd.DataFrame, p: int=14) -> float:
    try:
        rsi_series = compute_rsi_series(df, p)
        if rsi_series.empty:
            return 50.0
        return round(rsi_series.iloc[-1], 1)
    except Exception:
        return 50.0

def calculate_macd(df: pd.DataFrame) -> float:
    if len(df) < 27:
        return 0.0
    e12 = df["close"].ewm(span=12).mean()
    e26 = df["close"].ewm(span=26).mean()
    return round((e12 - e26).iloc[-1], 6)

def calculate_demarker(df: pd.DataFrame, p: int=14) -> float:
    if len(df) < p + 2:
        return 0.5
    h, l = df["high"].values, df["low"].values
    de_max = pd.Series([max(h[i] - h[i - 1], 0) for i in range(1, len(df))]).rolling(p).mean().iloc[-1]
    de_min = pd.Series([max(l[i - 1] - l[i], 0) for i in range(1, len(df))]).rolling(p).mean().iloc[-1]
    return round(de_max / (de_max + de_min + 1e-10), 3)

def calculate_bollinger_bands(df: pd.DataFrame, p: int=20, s: int=2):
    if len(df) < p:
        return None, None, None
    sma = df["close"].rolling(p).mean()
    std = df["close"].rolling(p).std()
    return sma.iloc[-1] + std.iloc[-1] * s, sma.iloc[-1], sma.iloc[-1] - std.iloc[-1] * s

def calculate_vwap_daily(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    # History can arrive with a RangeIndex in tests/offline mode. Normalize it
    # locally instead of making every caller construct a DatetimeIndex.
    work = df
    try:
        if not isinstance(work.index, pd.DatetimeIndex):
            work = work.copy()
            work.index = pd.date_range(
                end=pd.Timestamp.now(), periods=len(work), freq="min"
            )
    except Exception:
        return None
    today = datetime.now().date()
    df_today = work[work.index.date == today]
    if df_today.empty:
        return None
    tp = (df_today["high"] + df_today["low"] + df_today["close"]) / 3
    denom = df_today["volume"].sum()
    if denom == 0:
        return None
    return (tp * df_today["volume"]).sum() / denom

# NEW: VWAP with Standard Deviation Bands
def calculate_vwap_bands(df: pd.DataFrame, anchor="D"):
    """Calculate anchored VWAP with std bands."""
    try:
        if df.empty or len(df) < 20:
            return None, None, None
        tp = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
        variance = ((tp - vwap) ** 2 * df["volume"]).cumsum() / df["volume"].cumsum()
        std = np.sqrt(variance)
        return vwap.iloc[-1] + std.iloc[-1] * 2, vwap.iloc[-1], vwap.iloc[-1] - std.iloc[-1] * 2
    except Exception:
        return None, None, None

# NEW: Volume Profile (simple approximation)
def calculate_volume_profile(df: pd.DataFrame, bins: int=24):
    try:
        if df.empty or len(df) < bins:
            return None, None, None
        price_range = df["high"].max() - df["low"].min()
        if price_range == 0:
            return None, None, None
        bin_size = price_range / bins
        poc_price = None
        poc_vol = 0
        for i in range(bins):
            low_bin = df["low"].min() + i * bin_size
            high_bin = low_bin + bin_size
            mask = (df["close"] >= low_bin) & (df["close"] < high_bin)
            vol = df.loc[mask, "volume"].sum()
            if vol > poc_vol:
                poc_vol = vol
                poc_price = (low_bin + high_bin) / 2
        return poc_price, poc_vol, bins
    except Exception:
        return None, None, None

# NEW: Adaptive RSI (using Kaufman Efficiency Ratio)
def calculate_adaptive_rsi(df: pd.DataFrame, period: int=14):
    try:
        if len(df) < period * 2:
            return calculate_rsi(df, period)
        change = df["close"].diff().abs()
        volatility = change.rolling(period).sum()
        er = change.iloc[-period:].sum() / (volatility.iloc[-1] + 1e-10)
        fast = 2 / (2 + 1)
        slow = 2 / (30 + 1)
        sc = (er * (fast - slow) + slow) ** 2
        # Simplified adaptive RSI using smoothed gains/losses
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=sc, adjust=False).mean()
        avg_loss = loss.ewm(alpha=sc, adjust=False).mean()
        rs = avg_gain.iloc[-1] / (avg_loss.iloc[-1] + 1e-10)
        return 100 - (100 / (1 + rs))
    except Exception:
        return calculate_rsi(df, period)

# NEW: Keltner Channels
def calculate_keltner_channels(df: pd.DataFrame, ema_period: int=20, atr_period: int=14, mult: float=2.0):
    try:
        if len(df) < ema_period + atr_period:
            return None, None, None
        ema = df["close"].ewm(span=ema_period).mean()
        atr_val = calculate_atr(df, atr_period)
        mid = ema.iloc[-1]
        return mid + mult * atr_val, mid, mid - mult * atr_val
    except Exception:
        return None, None, None

# NEW: Donchian Channels
def calculate_donchian_channels(df: pd.DataFrame, period: int=20):
    try:
        if len(df) < period:
            return None, None, None
        upper = df["high"].rolling(period).max().iloc[-1]
        lower = df["low"].rolling(period).min().iloc[-1]
        mid = (upper + lower) / 2
        return upper, mid, lower
    except Exception:
        return None, None, None

# NEW: Ichimoku Cloud
def calculate_ichimoku(df: pd.DataFrame):
    try:
        if len(df) < 52:
            return None, None, None, None, None
        tenkan = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
        kijun = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2).shift(26)
        chikou = df["close"].shift(-26)
        return tenkan.iloc[-1], kijun.iloc[-1], senkou_a.iloc[-1], senkou_b.iloc[-1], chikou.iloc[-1]
    except Exception:
        return None, None, None, None, None

# NEW: Divergence Detection
def detect_divergence(price_series: pd.Series, indicator_series: pd.Series, lookback: int=20):
    """
    Detects bullish and bearish divergence.
    Returns: (bullish_div, bearish_div)
    """
    try:
        if len(price_series) < lookback or len(indicator_series) < lookback:
            return False, False
        p = price_series.iloc[-lookback:]
        i = indicator_series.iloc[-lookback:]
        # Bullish: price lower low, indicator higher low
        price_ll = p.idxmin()
        ind_at_price_ll = i.loc[price_ll]
        # Check if there's a previous low before price_ll
        prev_p = price_series.iloc[-lookback*2:-lookback]
        prev_i = indicator_series.iloc[-lookback*2:-lookback]
        if len(prev_p) < 5:
            return False, False
        prev_ll = prev_p.idxmin()
        prev_ind_ll = prev_i.loc[prev_ll]
        bullish = (p.loc[price_ll] < prev_p.loc[prev_ll]) and (ind_at_price_ll > prev_ind_ll)
        # Bearish: price higher high, indicator lower high
        price_hh = p.idxmax()
        ind_at_price_hh = i.loc[price_hh]
        prev_hh = prev_p.idxmax()
        prev_ind_hh = prev_i.loc[prev_hh]
        bearish = (p.loc[price_hh] > prev_p.loc[prev_hh]) and (ind_at_price_hh < prev_ind_hh)
        return bullish, bearish
    except Exception:
        return False, False

# NEW: ATR
def calculate_atr(df: pd.DataFrame, period: int=14) -> float:
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if not pd.isna(atr) else 0.0
    except Exception:
        return 0.0

# NEW: ADX
def calculate_adx(df: pd.DataFrame, period: int=14) -> float:
    if len(df) < period + 10:
        return 0.0
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().fillna(0)
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9))
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        val = 0.0
        if not adx.empty and not pd.isna(adx.iloc[-1]):
            val = round(adx.iloc[-1], 1)
        return val
    except Exception:
        return 0.0
# ---------------- Market Structure (Smart Money Concepts) ----------------
def detect_swing_points(df: pd.DataFrame, left: int=SWING_LOOKBACK, right: int=SWING_LOOKBACK):
    """
    Detects swing highs and lows.
    Returns: (swing_highs_idx, swing_lows_idx)
    """
    try:
        if len(df) < left + right + 1:
            return [], []
        highs = df["high"].values
        lows = df["low"].values
        swing_highs = []
        swing_lows = []
        for i in range(left, len(df) - right):
            # Swing High
            if all(highs[i] >= highs[i - j] for j in range(1, left + 1)) and all(highs[i] >= highs[i + j] for j in range(1, right + 1)):
                swing_highs.append(i)
            # Swing Low
            if all(lows[i] <= lows[i - j] for j in range(1, left + 1)) and all(lows[i] <= lows[i + j] for j in range(1, right + 1)):
                swing_lows.append(i)
        return swing_highs, swing_lows
    except Exception:
        return [], []

def detect_bos_choch(df: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]):
    """
    Detects Break of Structure (BOS) and Change of Character (CHoCH).
    Returns dict with keys: bos_bullish, bos_bearish, choch_bullish, choch_bearish
    """
    try:
        result = {"bos_bullish": False, "bos_bearish": False, "choch_bullish": False, "choch_bearish": False, "last_swing_high": None, "last_swing_low": None}
        if not swing_highs or not swing_lows:
            return result
        last_sh = swing_highs[-1]
        last_sl = swing_lows[-1]
        result["last_swing_high"] = df["high"].iloc[last_sh]
        result["last_swing_low"] = df["low"].iloc[last_sl]
        close = df["close"].iloc[-1]
        # BOS: Close beyond last swing point in trend direction
        if close > result["last_swing_high"]:
            result["bos_bullish"] = True
        if close < result["last_swing_low"]:
            result["bos_bearish"] = True
        # CHoCH: Close beyond last swing point against prior trend (simplified)
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            prev_sh = df["high"].iloc[swing_highs[-2]]
            prev_sl = df["low"].iloc[swing_lows[-2]]
            if close < prev_sl and close > result["last_swing_low"]:
                result["choch_bearish"] = True
            if close > prev_sh and close < result["last_swing_high"]:
                result["choch_bullish"] = True
        return result
    except Exception:
        return {"bos_bullish": False, "bos_bearish": False, "choch_bullish": False, "choch_bearish": False, "last_swing_high": None, "last_swing_low": None}

def detect_order_blocks(df: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]):
    """
    Detects bullish and bearish order blocks.
    Returns: (bullish_ob_list, bearish_ob_list)
    """
    try:
        bullish_obs = []
        bearish_obs = []
        if not swing_highs or not swing_lows:
            return bullish_obs, bearish_obs
        # Bearish OB: Before a swing high, the last bullish candle before the move up
        for sh in swing_highs[-3:]:
            if sh < OB_BARS + 1:
                continue
            window = df.iloc[sh - OB_BARS:sh]
            # Find the candle with highest volume in window before the high
            max_vol_idx = window["volume"].idxmax()
            if max_vol_idx in window.index:
                row = df.loc[max_vol_idx]
                bearish_obs.append({"high": row["high"], "low": row["low"], "index": max_vol_idx, "type": "bearish"})
        # Bullish OB: Before a swing low, the last bearish candle before the move down
        for sl in swing_lows[-3:]:
            if sl < OB_BARS + 1:
                continue
            window = df.iloc[sl - OB_BARS:sl]
            max_vol_idx = window["volume"].idxmax()
            if max_vol_idx in window.index:
                row = df.loc[max_vol_idx]
                bullish_obs.append({"high": row["high"], "low": row["low"], "index": max_vol_idx, "type": "bullish"})
        return bullish_obs, bearish_obs
    except Exception:
        return [], []

def detect_fvg(df: pd.DataFrame):
    """
    Detects Fair Value Gaps (3-candle imbalance).
    Returns: (bullish_fvgs, bearish_fvgs)
    """
    try:
        bullish = []
        bearish = []
        if len(df) < 10:
            return bullish, bearish
        for i in range(2, len(df)):
            c1 = df.iloc[i - 2]
            c2 = df.iloc[i - 1]
            c3 = df.iloc[i]
            # Bullish FVG: c1 high < c3 low
            if c1["high"] < c3["low"]:
                gap = c3["low"] - c1["high"]
                gap_pct = gap / c1["close"] * 100
                if gap_pct >= FVG_MIN_GAP_PCT:
                    bullish.append({"top": c3["low"], "bottom": c1["high"], "index": i, "gap_pct": gap_pct})
            # Bearish FVG: c1 low > c3 high
            if c1["low"] > c3["high"]:
                gap = c1["low"] - c3["high"]
                gap_pct = gap / c1["close"] * 100
                if gap_pct >= FVG_MIN_GAP_PCT:
                    bearish.append({"top": c1["low"], "bottom": c3["high"], "index": i, "gap_pct": gap_pct})
        return bullish, bearish
    except Exception:
        return [], []

def detect_liquidity_sweeps(df: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]):
    """
    Detects liquidity sweeps (stop hunts) beyond swing points.
    Returns: (sweep_high, sweep_low)
    """
    try:
        if not swing_highs or not swing_lows or len(df) < 5:
            return False, False
        last_sh = df["high"].iloc[swing_highs[-1]]
        last_sl = df["low"].iloc[swing_lows[-1]]
        recent = df.iloc[-3:]
        # Sweep high: wick above last swing high but close below
        sweep_high = any((r["high"] > last_sh * (1 + LIQUIDITY_SWEEP_PCT/100) and r["close"] < last_sh) for _, r in recent.iterrows())
        # Sweep low: wick below last swing low but close above
        sweep_low = any((r["low"] < last_sl * (1 - LIQUIDITY_SWEEP_PCT/100) and r["close"] > last_sl) for _, r in recent.iterrows())
        return sweep_high, sweep_low
    except Exception:
        return False, False

# ---------------- Microstructure ----------------
def calculate_orderbook_imbalance(sym: str, depth: int=ORDERBOOK_DEPTH_LEVELS) -> float:
    """
    Enhanced order book imbalance using depth-weighted metric.
    Returns: -1.0 (all ask) to +1.0 (all bid)
    FIX 3: Never prune symbols here; return 0.0 on failure.
    """
    try:
        ok, ob = robust_orderbook(sym)
        if not ok or not isinstance(ob, dict):
            return 0.0
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            return 0.0
        def extract_levels(levels, n):
            out = []
            for lvl in levels[:n]:
                if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    out.append((safe_float(lvl[0]), safe_float(lvl[1])))
                elif isinstance(lvl, dict):
                    out.append((safe_float(lvl.get("price", lvl.get("p", 0))), safe_float(lvl.get("quantity", lvl.get("q", 0)))))
            return out
        bid_levels = extract_levels(bids, depth)
        ask_levels = extract_levels(asks, depth)
        if not bid_levels or not ask_levels:
            return 0.0
        # Depth-weighted: closer to mid gets more weight
        mid = (bid_levels[0][0] + ask_levels[0][0]) / 2
        bid_weighted = sum(q / (1 + abs(p - mid)/mid) for p, q in bid_levels)
        ask_weighted = sum(q / (1 + abs(p - mid)/mid) for p, q in ask_levels)
        total = bid_weighted + ask_weighted
        if total == 0:
            return 0.0
        return (bid_weighted - ask_weighted) / total
    except Exception:
        return 0.0

def estimate_vpin(df: pd.DataFrame, bars: int=VPIN_BARS) -> float:
    """
    Volume-synchronized Probability of Informed Trading (VPIN).
    Simplified using absolute returns per volume bucket.
    """
    try:
        if len(df) < bars * 2:
            return 0.0
        vol_bucket = df["volume"].rolling(bars).sum().iloc[-1] / bars
        if vol_bucket <= 0:
            return 0.0
        recent = df.iloc[-bars:]
        returns = recent["close"].pct_change().abs()
        vpin = (returns * recent["volume"]).sum() / (vol_bucket * bars)
        return min(1.0, max(0.0, vpin))
    except Exception:
        return 0.0

# ---------------- Multi-Timeframe Analysis ----------------
def _make_signal_signature(score: float, talaye: bool, rule_ids) -> str:
    try:
        rules = ",".join(sorted(map(str, rule_ids or [])))
        return f"{round(float(score),2)}|{int(bool(talaye))}|{rules}"
    except Exception:
        return str(time.time())

def calculate_mtf_alignment(sym: str) -> float:
    """Bounded MTF calculation. Missing timeframes are ignored, never fatal."""
    try:
        key = canonical(sym)
        cache_key = f"mtf_{key}"
        cached = cache.get(cache_key)
        if cached and time.time() - cached.get("t", 0) < 300:
            return float(cached.get("score", 0.0))

        specs = [(15, 0.25), (60, 0.35), (240, 0.40)]
        scores, weights = [], []
        for tf, w in specs:
            df = get_candles_cached(key, tf, n=80, valid_symbols_map=valid_symbols_map)
            if df is None or df.empty or len(df) < 30:
                continue
            close = df["close"]
            ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
            ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
            ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            p = float(close.iloc[-1])
            if p > ema9 > ema21 > ema50:
                s = 1.0
            elif p > ema9 > ema21:
                s = 0.7
            elif p > ema9:
                s = 0.4
            elif p < ema9 < ema21 < ema50:
                s = -1.0
            elif p < ema9 < ema21:
                s = -0.7
            elif p < ema9:
                s = -0.4
            else:
                s = 0.0
            scores.append(s)
            weights.append(w)

        if not scores:
            return 0.5

        weighted = sum(s * w for s, w in zip(scores, weights))
        result = min(1.0, max(0.0, (weighted / max(sum(weights), 1e-9) + 1) / 2))
        cache[cache_key] = {"score": result, "t": time.time()}
        _mtf_alignment_cache[key] = result
        return result
    except Exception:
        return 0.5

# ---------------- Feature Engineering ----------------
def build_advanced_features(sym: str, resolution: int=60) -> Optional[Dict[str, float]]:
    """
    Builds comprehensive feature set for ML models.
    """
    try:
        df = get_candles_cached(sym, resolution, n=500, valid_symbols_map=valid_symbols_map)
        if df is None or df.empty or len(df) < 50:
            return None

        feats = {}
        close = df["close"].iloc[-1]
        feats["close"] = float(close)

        # Returns
        feats["ret_1"] = float((close / df["close"].iloc[-2] - 1.0) * 100.0) if len(df) >= 2 else 0.0
        feats["ret_4"] = float((close / df["close"].iloc[-5] - 1.0) * 100.0) if len(df) >= 5 else 0.0
        feats["ret_24"] = float((close / df["close"].iloc[-25] - 1.0) * 100.0) if len(df) >= 25 else 0.0

        # Volatility
        feats["atr_14"] = float(calculate_atr(df, 14))
        feats["atr_14_pct"] = float(feats["atr_14"] / close * 100)

        # Trend
        feats["adx_14"] = float(calculate_adx(df, 14))
        feats["rsi_14"] = float(calculate_rsi(df, 14))
        feats["rsi_adaptive"] = float(calculate_adaptive_rsi(df, 14))
        feats["macd"] = float(calculate_macd(df))

        # Bollinger
        ub, ma, lb = calculate_bollinger_bands(df)
        if ma and ma != 0 and ub is not None and lb is not None:
            feats["bb_width"] = float((ub - lb) / ma)
            feats["bb_position"] = float((close - lb) / (ub - lb)) if (ub - lb) > 0 else 0.5
        else:
            feats["bb_width"] = 0.0
            feats["bb_position"] = 0.5

        # Volume
        feats["vol_z"] = float(volume_zscore(df, lookback=100))
        feats["vol_ratio_10"] = float(get_volume_ratio_from_candles(df, recent_bars=10, prev_bars=100))

        # VWAP
        vwap = calculate_vwap_daily(df)
        feats["vwap_diff"] = float((close - vwap) / vwap) if vwap and vwap != 0 else 0.0

        # Keltner & Donchian
        kc = calculate_keltner_channels(df)
        if kc[0] is not None:
            feats["kc_position"] = float((close - kc[2]) / (kc[0] - kc[2])) if (kc[0] - kc[2]) > 0 else 0.5
        else:
            feats["kc_position"] = 0.5

        dc = calculate_donchian_channels(df)
        if dc[0] is not None:
            feats["dc_position"] = float((close - dc[2]) / (dc[0] - dc[2])) if (dc[0] - dc[2]) > 0 else 0.5
        else:
            feats["dc_position"] = 0.5

        # Ichimoku
        ichi = calculate_ichimoku(df)
        if ichi[0] is not None:
            feats["ichi_tenkan"] = float((close - ichi[1]) / close * 100)
            feats["ichi_cloud"] = float((ichi[2] - ichi[3]) / close * 100) if ichi[2] and ichi[3] else 0.0
        else:
            feats["ichi_tenkan"] = 0.0
            feats["ichi_cloud"] = 0.0

        # Divergence
        rsi_series = compute_rsi_series(df, 14)
        bull_div, bear_div = detect_divergence(df["close"], rsi_series)
        feats["rsi_bull_div"] = 1.0 if bull_div else 0.0
        feats["rsi_bear_div"] = 1.0 if bear_div else 0.0

        # Microstructure
        feats["delta"] = float(last_delta_cache.get(canonical(sym), 0.0))
        feats["ob_imbalance"] = float(calculate_orderbook_imbalance(sym))
        feats["vpin"] = float(estimate_vpin(df))

        # Market Structure
        sh, sl = detect_swing_points(df)
        bos = detect_bos_choch(df, sh, sl)
        feats["bos_bullish"] = 1.0 if bos["bos_bullish"] else 0.0
        feats["bos_bearish"] = 1.0 if bos["bos_bearish"] else 0.0
        feats["choch_bullish"] = 1.0 if bos["choch_bullish"] else 0.0
        feats["choch_bearish"] = 1.0 if bos["choch_bearish"] else 0.0

        bullish_ob, bearish_ob = detect_order_blocks(df, sh, sl)
        feats["bullish_ob_near"] = 1.0 if any(abs(close - ob["low"])/close < 0.01 for ob in bullish_ob[-2:]) else 0.0
        feats["bearish_ob_near"] = 1.0 if any(abs(close - ob["high"])/close < 0.01 for ob in bearish_ob[-2:]) else 0.0

        fvg_bull, fvg_bear = detect_fvg(df)
        feats["fvg_bull_near"] = 1.0 if any(close > f["bottom"] and close < f["top"] for f in fvg_bull[-3:]) else 0.0
        feats["fvg_bear_near"] = 1.0 if any(close > f["bottom"] and close < f["top"] for f in fvg_bear[-3:]) else 0.0

        sweep_high, sweep_low = detect_liquidity_sweeps(df, sh, sl)
        feats["sweep_high"] = 1.0 if sweep_high else 0.0
        feats["sweep_low"] = 1.0 if sweep_low else 0.0

        # Multi-timeframe
        feats["mtf_align"] = float(calculate_mtf_alignment(sym))

        # Slope
        feats["slope_sma20"] = float(slope_of_series(df["close"].rolling(20).mean().dropna(), periods=8))
        feats["slope_sma50"] = float(slope_of_series(df["close"].rolling(50).mean().dropna(), periods=8))

        # Volume Profile
        poc, poc_vol, _ = calculate_volume_profile(df)
        if poc is not None:
            feats["poc_distance"] = float((close - poc) / close * 100)
        else:
            feats["poc_distance"] = 0.0

        return feats
    except Exception:
        logging.debug("build_advanced_features error: %s", traceback.format_exc())
        return None

def volume_zscore(df: pd.DataFrame, lookback: int=100) -> float:
    try:
        v = df["volume"].tail(lookback)
        if len(v) < 10:
            v = df["volume"]
        if v.empty:
            return 0.0
        mean = v.mean()
        std = v.std() if v.std() > 0 else 1.0
        return float((v.iloc[-1] - mean) / std)
    except Exception:
        return 0.0

def slope_of_series(s: pd.Series, periods: int=5) -> float:
    try:
        y = s.dropna().values[-periods:]
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m)
    except Exception:
        return 0.0

def get_volume_ratio_from_candles(dfp: pd.DataFrame, recent_bars: int=3, prev_bars: int=24):
    try:
        if dfp is None or dfp.empty:
            return 0.0
        if recent_bars <= 0:
            recent_bars = 1
        total = len(dfp)
        if total <= recent_bars:
            return 0.0
        recent_vol = dfp["volume"].tail(recent_bars).sum()
        avail_prev = total - recent_bars
        use_prev = min(prev_bars, avail_prev)
        if use_prev <= 0:
            return 0.0
        prev = dfp["volume"].iloc[-(recent_bars + use_prev):-recent_bars] if use_prev > 0 else pd.Series([])
        prev_avg = prev.mean() if not prev.empty else 0.0
        if prev_avg <= 0:
            return 0.0
        return round(recent_vol / prev_avg, 2)
    except Exception:
        return 0.0

# ---------------- ML Models ----------------
FEATURE_COLS = [
    "rsi_14", "rsi_adaptive", "macd", "adx_14", "atr_14_pct", "bb_width", "bb_position",
    "vol_z", "vol_ratio_10", "vwap_diff", "kc_position", "dc_position", "ichi_tenkan",
    "ichi_cloud", "rsi_bull_div", "rsi_bear_div", "delta", "ob_imbalance", "vpin",
    "bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish", "bullish_ob_near",
    "bearish_ob_near", "fvg_bull_near", "fvg_bear_near", "sweep_high", "sweep_low",
    "mtf_align", "slope_sma20", "slope_sma50", "poc_distance", "ret_1", "ret_4", "ret_24"
]

def load_ml_ensemble():
    """Load or initialize ML ensemble models."""
    global _ml_models_loaded, _ml_ensemble, _ml_scaler, _ml_feature_importance, _ml_last_train_time
    if _ml_models_loaded:
        return
    try:
        os.makedirs(ML_ARTIFACT_DIR, exist_ok=True)
        _ml_ensemble = {}

        # Load scaler
        scaler_path = os.path.join(ML_ARTIFACT_DIR, "scaler.joblib")
        if os.path.exists(scaler_path) and _ML_AVAILABLE.get("sklearn"):
            _ml_scaler = joblib.load(scaler_path)
            logging.info("Loaded scaler from %s", scaler_path)

        # Load Random Forest
        rf_clf_path = os.path.join(ML_ARTIFACT_DIR, "rf_clf.joblib")
        rf_reg_path = os.path.join(ML_ARTIFACT_DIR, "rf_reg.joblib")
        if os.path.exists(rf_clf_path) and _ML_AVAILABLE.get("sklearn"):
            _ml_ensemble["rf_clf"] = joblib.load(rf_clf_path)
            logging.info("Loaded RF classifier")
        if os.path.exists(rf_reg_path) and _ML_AVAILABLE.get("sklearn"):
            _ml_ensemble["rf_reg"] = joblib.load(rf_reg_path)
            logging.info("Loaded RF regressor")

        # Load XGBoost
        xgb_clf_path = os.path.join(ML_ARTIFACT_DIR, "xgb_clf.joblib")
        xgb_reg_path = os.path.join(ML_ARTIFACT_DIR, "xgb_reg.joblib")
        if os.path.exists(xgb_clf_path) and _ML_AVAILABLE.get("xgboost"):
            _ml_ensemble["xgb_clf"] = joblib.load(xgb_clf_path)
            logging.info("Loaded XGB classifier")
        if os.path.exists(xgb_reg_path) and _ML_AVAILABLE.get("xgboost"):
            _ml_ensemble["xgb_reg"] = joblib.load(xgb_reg_path)
            logging.info("Loaded XGB regressor")

        # Load LightGBM
        lgb_clf_path = os.path.join(ML_ARTIFACT_DIR, "lgb_clf.joblib")
        lgb_reg_path = os.path.join(ML_ARTIFACT_DIR, "lgb_reg.joblib")
        if os.path.exists(lgb_clf_path) and _ML_AVAILABLE.get("lightgbm"):
            _ml_ensemble["lgb_clf"] = joblib.load(lgb_clf_path)
            logging.info("Loaded LGB classifier")
        if os.path.exists(lgb_reg_path) and _ML_AVAILABLE.get("lightgbm"):
            _ml_ensemble["lgb_reg"] = joblib.load(lgb_reg_path)
            logging.info("Loaded LGB regressor")

        # Load LSTM
        lstm_path = os.path.join(ML_ARTIFACT_DIR, "lstm_model.h5")
        if os.path.exists(lstm_path) and _ML_AVAILABLE.get("tensorflow"):
            _ml_ensemble["lstm"] = load_model(lstm_path)
            logging.info("Loaded LSTM model")

        # Load feature importance
        fi_path = os.path.join(ML_ARTIFACT_DIR, "feature_importance.json")
        if os.path.exists(fi_path):
            with open(fi_path, "r") as f:
                _ml_feature_importance = json.load(f)

        _ml_last_train_time = os.path.getmtime(scaler_path) if os.path.exists(scaler_path) else 0

    except Exception:
        logging.debug("load_ml_ensemble failed: %s", traceback.format_exc())
    finally:
        _ml_models_loaded = True

def predict_with_ensemble(sym: str) -> Dict[str, float]:
    """
    Predict using ensemble of models.
    Returns: {"prob_up": float, "expected_move": float, "model_confidence": float}
    """
    try:
        load_ml_ensemble()
        feats = build_advanced_features(sym, resolution=60)
        if feats is None:
            return {"prob_up": 0.5, "expected_move": 0.0, "model_confidence": 0.0}

        X = np.array([[feats.get(c, 0.0) for c in FEATURE_COLS]])

        prob_predictions = []
        move_predictions = []
        model_weights = []

        # Scale if scaler available
        Xs = X
        if _ml_scaler is not None and _ML_AVAILABLE.get("sklearn"):
            try:
                Xs = _ml_scaler.transform(X)
            except Exception:
                Xs = X

        # Random Forest
        if "rf_clf" in _ml_ensemble:
            try:
                if hasattr(_ml_ensemble["rf_clf"], "predict_proba"):
                    prob = float(_ml_ensemble["rf_clf"].predict_proba(Xs)[0, 1])
                else:
                    prob = float(_ml_ensemble["rf_clf"].predict(Xs)[0])
                prob_predictions.append(prob)
                model_weights.append(0.25)
            except Exception:
                pass
        if "rf_reg" in _ml_ensemble:
            try:
                move = float(_ml_ensemble["rf_reg"].predict(Xs)[0])
                move_predictions.append(move)
            except Exception:
                pass

        # XGBoost
        if "xgb_clf" in _ml_ensemble:
            try:
                if hasattr(_ml_ensemble["xgb_clf"], "predict_proba"):
                    prob = float(_ml_ensemble["xgb_clf"].predict_proba(Xs)[0, 1])
                else:
                    prob = float(_ml_ensemble["xgb_clf"].predict(Xs)[0])
                prob_predictions.append(prob)
                model_weights.append(0.30)
            except Exception:
                pass
        if "xgb_reg" in _ml_ensemble:
            try:
                move = float(_ml_ensemble["xgb_reg"].predict(Xs)[0])
                move_predictions.append(move)
            except Exception:
                pass

        # LightGBM
        if "lgb_clf" in _ml_ensemble:
            try:
                if hasattr(_ml_ensemble["lgb_clf"], "predict_proba"):
                    prob = float(_ml_ensemble["lgb_clf"].predict_proba(Xs)[0, 1])
                else:
                    prob = float(_ml_ensemble["lgb_clf"].predict(Xs)[0])
                prob_predictions.append(prob)
                model_weights.append(0.25)
            except Exception:
                pass
        if "lgb_reg" in _ml_ensemble:
            try:
                move = float(_ml_ensemble["lgb_reg"].predict(Xs)[0])
                move_predictions.append(move)
            except Exception:
                pass

        # LSTM (sequence-based)
        if "lstm" in _ml_ensemble:
            try:
                df = get_candles_cached(sym, 60, n=ML_SEQUENCE_LENGTH + 10, valid_symbols_map=valid_symbols_map)
                if df is not None and len(df) >= ML_SEQUENCE_LENGTH:
                    seq = df[["open", "high", "low", "close", "volume"]].iloc[-ML_SEQUENCE_LENGTH:].values
                    seq = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-10)
                    seq = np.expand_dims(seq, axis=0)
                    lstm_pred = _ml_ensemble["lstm"].predict(seq, verbose=0)
                    prob_predictions.append(float(lstm_pred[0][0]))
                    model_weights.append(0.20)
            except Exception:
                pass

        # Fallback to rule-based if no models
        if not prob_predictions:
            rule_move = rule_based_estimate_move_from_feats(feats)
            rule_prob = rule_signal_strength(feats)
            return {"prob_up": rule_prob, "expected_move": rule_move, "model_confidence": 0.3}

        # Weighted ensemble
        total_weight = sum(model_weights)
        if total_weight > 0:
            weights_norm = [w / total_weight for w in model_weights]
            prob_up = sum(p * w for p, w in zip(prob_predictions, weights_norm))
        else:
            prob_up = sum(prob_predictions) / len(prob_predictions)

        expected_move = sum(move_predictions) / len(move_predictions) if move_predictions else 0.0

        # Model confidence based on agreement (lower std = higher confidence)
        if len(prob_predictions) >= 2:
            agreement = 1.0 - min(1.0, np.std(prob_predictions) * 2)
            model_conf = agreement * min(1.0, len(prob_predictions) / 4.0)
        else:
            model_conf = 0.5

        prob_up = max(0.0, min(1.0, prob_up))
        expected_move = max(0.0, min(50.0, expected_move))

        return {"prob_up": prob_up, "expected_move": round(expected_move, 2), "model_confidence": round(model_conf, 2)}
    except Exception:
        logging.debug("predict_with_ensemble error: %s", traceback.format_exc())
        return {"prob_up": 0.5, "expected_move": 0.0, "model_confidence": 0.0}

def rule_based_estimate_move_from_feats(feats: dict) -> float:
    try:
        if not feats:
            return 0.5
        vol_factor = max(0.0, min(6.0, feats.get("vol_z", 0.0)))
        atr_pct = feats.get("atr_14_pct", 0.0)
        trend_bonus = max(0.0, min(5.0, abs(feats.get("slope_sma20", 0.0)) * 100.0))
        rsi = feats.get("rsi_14", 50.0)
        rsi_bonus = 0.0
        if rsi < 35:
            rsi_bonus = (35 - rsi) * 0.04
        elif rsi > 65:
            rsi_bonus = (rsi - 65) * 0.03
        mtf = feats.get("mtf_align", 0.5)
        mtf_bonus = abs(mtf - 0.5) * 4.0
        base = 0.2 + vol_factor * 0.6 + atr_pct * 0.6 + trend_bonus * 0.2 + rsi_bonus + mtf_bonus
        base = max(0.2, min(20.0, base))
        return round(base, 2)
    except Exception:
        return 0.5

def rule_signal_strength(feats: dict) -> float:
    try:
        if not feats:
            return 0.0
        s = 0.0
        s += min(1.0, max(0.0, feats.get("vol_z", 0.0) / 2.5))
        s += min(1.0, max(0.0, (feats.get("adx_14", 0.0) - 10)/40.0))
        s += min(1.0, max(0.0, abs(feats.get("slope_sma20",0.0))*10.0))
        s += feats.get("mtf_align", 0.5)
        return max(0.0, min(1.0, s/4.0))
    except Exception:
        return 0.0

# ---------------- Risk Management ----------------
def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly fraction. Returns 0 if negative expectancy.
    """
    try:
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss  # odds
        q = 1 - win_rate
        kelly = (win_rate * b - q) / b if b != 0 else 0.0
        return max(0.0, kelly) * KELLY_FRACTION  # Use fractional Kelly
    except Exception:
        return 0.0

def optimal_position_size(capital: float, risk_pct: float, entry: float, stop: float, leverage: float=LEVERAGE) -> float:
    """
    Calculate optimal position size based on risk.
    """
    try:
        if entry <= 0 or stop <= 0 or capital <= 0:
            return 0.0
        risk_amount = capital * (risk_pct / 100.0)
        price_diff = abs(entry - stop)
        if price_diff == 0:
            return 0.0
        raw_size = risk_amount / price_diff
        leveraged_size = raw_size / leverage
        return max(0.0, leveraged_size)
    except Exception:
        return 0.0

def monte_carlo_simulation(trades: List[Dict], n_sims: int=MONTE_CARLO_SIMS, initial_capital: float=1000.0):
    """
    Run Monte Carlo simulation on trade history.
    Returns: dict with statistics
    """
    try:
        if not trades:
            return {"median_final": initial_capital, "max_dd": 0.0, "prob_ruin": 0.0, "sharpe": 0.0}
        returns = [t.get("final_pct", 0.0) for t in trades if t.get("final_pct") is not None]
        if not returns:
            return {"median_final": initial_capital, "max_dd": 0.0, "prob_ruin": 0.0, "sharpe": 0.0}

        final_equities = []
        max_drawdowns = []
        ruin_count = 0

        for _ in range(n_sims):
            equity = initial_capital
            peak = equity
            max_dd = 0.0
            random_returns = random.choices(returns, k=len(returns))
            for ret in random_returns:
                equity *= (1 + ret / 100.0)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd
                if equity <= initial_capital * 0.1:  # 90% loss = ruin
                    ruin_count += 1
                    break
            final_equities.append(equity)
            max_drawdowns.append(max_dd)

        final_equities.sort()
        max_drawdowns.sort()

        return {
            "median_final": final_equities[len(final_equities)//2],
            "worst_final": final_equities[int(n_sims * 0.05)],
            "best_final": final_equities[int(n_sims * 0.95)],
            "median_max_dd": max_drawdowns[len(max_drawdowns)//2],
            "worst_max_dd": max_drawdowns[int(n_sims * 0.95)],
            "prob_ruin": ruin_count / n_sims,
            "sharpe": np.mean(returns) / (np.std(returns) + 1e-10) if returns else 0.0
        }
    except Exception:
        return {"median_final": initial_capital, "max_dd": 0.0, "prob_ruin": 0.0, "sharpe": 0.0}

def update_drawdown_metrics():
    """Update global drawdown tracking."""
    global _max_equity, _current_drawdown
    try:
        with state_lock:
            if _equity_curve:
                current = _equity_curve[-1]
                if current > _max_equity:
                    _max_equity = current
                _current_drawdown = (_max_equity - current) / _max_equity * 100.0
    except Exception:
        pass

def should_pause_trading() -> bool:
    """Check if trading should pause due to excessive drawdown."""
    try:
        update_drawdown_metrics()
        return _current_drawdown >= MAX_DRAWDOWN_PCT
    except Exception:
        return False

# ---------------- ML Training ----------------
def _feats_from_window_advanced(win: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Build features from historical window (no data leakage)."""
    try:
        if win is None or len(win) < 50:
            return None
        feats = {}
        close_last = float(win["close"].iloc[-1])
        feats["close"] = close_last
        feats["ret_1"] = float((close_last / win["close"].iloc[-2] - 1.0) * 100.0) if len(win) >= 2 else 0.0
        feats["ret_4"] = float((close_last / win["close"].iloc[-5] - 1.0) * 100.0) if len(win) >= 5 else 0.0
        feats["ret_24"] = float((close_last / win["close"].iloc[-25] - 1.0) * 100.0) if len(win) >= 25 else 0.0
        feats["atr_14"] = float(calculate_atr(win, 14))
        feats["atr_14_pct"] = float(feats["atr_14"] / close_last * 100)
        feats["adx_14"] = float(calculate_adx(win, 14))
        feats["rsi_14"] = float(calculate_rsi(win, 14))
        feats["rsi_adaptive"] = float(calculate_adaptive_rsi(win, 14))
        feats["macd"] = float(calculate_macd(win))
        ub, ma, lb = calculate_bollinger_bands(win)
        if ma and ma != 0 and ub is not None and lb is not None:
            feats["bb_width"] = float((ub - lb) / ma)
            feats["bb_position"] = float((close_last - lb) / (ub - lb)) if (ub - lb) > 0 else 0.5
        else:
            feats["bb_width"] = 0.0
            feats["bb_position"] = 0.5
        feats["vol_z"] = float(volume_zscore(win, 100))
        feats["vol_ratio_10"] = float(get_volume_ratio_from_candles(win, 10, 100))
        # VWAP rolling 24
        try:
            tp = (win["high"] + win["low"] + win["close"]) / 3.0
            vsum = float(win["volume"].tail(24).sum())
            if vsum > 0:
                vwap_roll = float((tp * win["volume"]).tail(24).sum() / vsum)
                feats["vwap_diff"] = float((close_last - vwap_roll) / vwap_roll) if vwap_roll != 0 else 0.0
            else:
                feats["vwap_diff"] = 0.0
        except Exception:
            feats["vwap_diff"] = 0.0
        kc = calculate_keltner_channels(win)
        if kc[0] is not None:
            feats["kc_position"] = float((close_last - kc[2]) / (kc[0] - kc[2])) if (kc[0] - kc[2]) > 0 else 0.5
        else:
            feats["kc_position"] = 0.5
        dc = calculate_donchian_channels(win)
        if dc[0] is not None:
            feats["dc_position"] = float((close_last - dc[2]) / (dc[0] - dc[2])) if (dc[0] - dc[2]) > 0 else 0.5
        else:
            feats["dc_position"] = 0.5
        ichi = calculate_ichimoku(win)
        if ichi[0] is not None:
            feats["ichi_tenkan"] = float((close_last - ichi[1]) / close_last * 100)
            feats["ichi_cloud"] = float((ichi[2] - ichi[3]) / close_last * 100) if ichi[2] and ichi[3] else 0.0
        else:
            feats["ichi_tenkan"] = 0.0
            feats["ichi_cloud"] = 0.0
        rsi_series = compute_rsi_series(win, 14)
        bull_div, bear_div = detect_divergence(win["close"], rsi_series)
        feats["rsi_bull_div"] = 1.0 if bull_div else 0.0
        feats["rsi_bear_div"] = 1.0 if bear_div else 0.0
        feats["delta"] = 0.0
        feats["ob_imbalance"] = 0.0
        feats["vpin"] = float(estimate_vpin(win))
        sh, sl = detect_swing_points(win)
        bos = detect_bos_choch(win, sh, sl)
        feats["bos_bullish"] = 1.0 if bos["bos_bullish"] else 0.0
        feats["bos_bearish"] = 1.0 if bos["bos_bearish"] else 0.0
        feats["choch_bullish"] = 1.0 if bos["choch_bullish"] else 0.0
        feats["choch_bearish"] = 1.0 if bos["choch_bearish"] else 0.0
        bullish_ob, bearish_ob = detect_order_blocks(win, sh, sl)
        feats["bullish_ob_near"] = 1.0 if any(abs(close_last - ob["low"])/close_last < 0.01 for ob in bullish_ob[-2:]) else 0.0
        feats["bearish_ob_near"] = 1.0 if any(abs(close_last - ob["high"])/close_last < 0.01 for ob in bearish_ob[-2:]) else 0.0
        fvg_bull, fvg_bear = detect_fvg(win)
        feats["fvg_bull_near"] = 1.0 if any(close_last > f["bottom"] and close_last < f["top"] for f in fvg_bull[-3:]) else 0.0
        feats["fvg_bear_near"] = 1.0 if any(close_last > f["bottom"] and close_last < f["top"] for f in fvg_bear[-3:]) else 0.0
        sweep_high, sweep_low = detect_liquidity_sweeps(win, sh, sl)
        feats["sweep_high"] = 1.0 if sweep_high else 0.0
        feats["sweep_low"] = 1.0 if sweep_low else 0.0
        feats["mtf_align"] = 0.5  # Cannot compute MTF from single window efficiently
        feats["slope_sma20"] = float(slope_of_series(win["close"].rolling(20).mean().dropna(), 8))
        feats["slope_sma50"] = float(slope_of_series(win["close"].rolling(50).mean().dropna(), 8))
        poc, _, _ = calculate_volume_profile(win)
        feats["poc_distance"] = float((close_last - poc) / close_last * 100) if poc else 0.0
        return feats
    except Exception:
        return None

def ml_build_dataset_advanced(symbols: List[str], resolution: int=60, bars: int=1500, min_history: int=100):
    """Build training dataset with advanced features."""
    try:
        all_rows = []
        for sym in symbols:
            try:
                df = get_candles_cached(sym, resolution, n=bars, valid_symbols_map=valid_symbols_map)
                if df is None or df.empty or len(df) < min_history + 6:
                    continue
                horizon = 2
                threshold_pct = 0.8
                closes = df["close"].astype(float).values
                labels_up = [0] * len(closes)
                max_fut = [0.0] * len(closes)
                for i in range(len(closes) - horizon):
                    cur = closes[i]
                    future = closes[i+1:i+1+horizon]
                    if future.size == 0:
                        continue
                    max_ret = max(((f / cur - 1.0) * 100.0) for f in future)
                    labels_up[i] = 1 if max_ret >= threshold_pct else 0
                    max_fut[i] = max_ret
                for i in range(min_history, len(df) - horizon - 1):
                    win = df.iloc[max(0, i - 240): i + 1]
                    feats = _feats_from_window_advanced(win)
                    if feats is None:
                        continue
                    feats["symbol"] = sym
                    feats["time"] = str(df.index[i])
                    feats["label"] = int(labels_up[i])
                    feats["target_move"] = float(max_fut[i])
                    all_rows.append(feats)
            except Exception:
                continue
        if not all_rows:
            return None
        return pd.DataFrame(all_rows)
    except Exception:
        return None

def ml_train_ensemble(symbols: List[str], out_dir: str=ML_ARTIFACT_DIR, resolution: int=60):
    """Train full ML ensemble and save artifacts."""
    if not _ML_AVAILABLE.get("sklearn"):
        logging.error("sklearn not available; cannot train")
        return
    os.makedirs(out_dir, exist_ok=True)
    logging.info("Starting ML ensemble training on %d symbols...", len(symbols))
    df_all = ml_build_dataset_advanced(symbols, resolution=resolution)
    if df_all is None or df_all.empty:
        logging.warning("No training data collected")
        return
    df_all = df_all.dropna()
    if df_all.empty:
        logging.warning("Dataset empty after dropna")
        return

    X = df_all[FEATURE_COLS].values
    y_clf = df_all["label"].values
    y_reg = df_all["target_move"].values

    # Scale
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    # Split
    X_train, X_test, yc_train, yc_test = train_test_split(X_s, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    _, _, yr_train, yr_test = train_test_split(X_s, y_reg, test_size=0.2, random_state=42)

    # Random Forest
    logging.info("Training Random Forest...")
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1, class_weight="balanced")
    rf_reg = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, yc_train)
    rf_reg.fit(X_train, yr_train)
    joblib.dump(rf_clf, os.path.join(out_dir, "rf_clf.joblib"))
    joblib.dump(rf_reg, os.path.join(out_dir, "rf_reg.joblib"))

    # XGBoost
    if _ML_AVAILABLE.get("xgboost"):
        logging.info("Training XGBoost...")
        xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric="logloss")
        xgb_reg = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)
        xgb_clf.fit(X_train, yc_train)
        xgb_reg.fit(X_train, yr_train)
        joblib.dump(xgb_clf, os.path.join(out_dir, "xgb_clf.joblib"))
        joblib.dump(xgb_reg, os.path.join(out_dir, "xgb_reg.joblib"))

    # LightGBM
    if _ML_AVAILABLE.get("lightgbm"):
        logging.info("Training LightGBM...")
        lgb_clf = lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, verbose=-1)
        lgb_reg = lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, verbose=-1)
        lgb_clf.fit(X_train, yc_train)
        lgb_reg.fit(X_train, yr_train)
        joblib.dump(lgb_clf, os.path.join(out_dir, "lgb_clf.joblib"))
        joblib.dump(lgb_reg, os.path.join(out_dir, "lgb_reg.joblib"))

    # LSTM
    if _ML_AVAILABLE.get("tensorflow"):
        logging.info("Training LSTM...")
        try:
            lstm_symbols = symbols[:min(5, len(symbols))]
            seq_data = []
            seq_labels = []
            for sym in lstm_symbols:
                df = get_candles_cached(sym, resolution, n=ML_SEQUENCE_LENGTH + 200, valid_symbols_map=valid_symbols_map)
                if df is not None and len(df) >= ML_SEQUENCE_LENGTH + 50:
                    for i in range(ML_SEQUENCE_LENGTH, len(df) - 2):
                        seq = df[["open", "high", "low", "close", "volume"]].iloc[i-ML_SEQUENCE_LENGTH:i].values
                        seq = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-10)
                        future_ret = (df["close"].iloc[i+2] / df["close"].iloc[i] - 1) * 100
                        seq_data.append(seq)
                        seq_labels.append(1.0 if future_ret >= 0.8 else 0.0)
            if seq_data:
                X_lstm = np.array(seq_data)
                y_lstm = np.array(seq_labels)
                split = int(0.8 * len(X_lstm))
                Xl_train, Xl_test = X_lstm[:split], X_lstm[split:]
                yl_train, yl_test = y_lstm[:split], y_lstm[split:]

                model = Sequential([
                    LSTM(ML_LSTM_UNITS, return_sequences=True, input_shape=(ML_SEQUENCE_LENGTH, 5)),
                    Dropout(ML_DROPOUT),
                    LSTM(ML_LSTM_UNITS // 2),
                    Dropout(ML_DROPOUT),
                    Dense(32, activation="relu"),
                    Dense(1, activation="sigmoid")
                ])
                model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
                es = EarlyStopping(patience=ML_PATIENCE, restore_best_weights=True)
                model.fit(Xl_train, yl_train, epochs=ML_EPOCHS, batch_size=ML_BATCH_SIZE, validation_split=0.1, callbacks=[es], verbose=0)
                model.save(os.path.join(out_dir, "lstm_model.h5"))
        except Exception:
            logging.debug("LSTM training failed: %s", traceback.format_exc())

    # Feature importance
    try:
        fi = dict(zip(FEATURE_COLS, rf_clf.feature_importances_.tolist()))
        fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
        with open(os.path.join(out_dir, "feature_importance.json"), "w") as f:
            json.dump(fi_sorted, f, indent=2)
    except Exception:
        pass

    # Evaluation
    try:
        preds = rf_clf.predict(X_test)
        logging.info("RF Classification Report:\n%s", classification_report(yc_test, preds))
        reg_preds = rf_reg.predict(X_test)
        logging.info("RF Regressor R2: %.4f", r2_score(yr_test, reg_preds))
    except Exception:
        pass

    logging.info("ML ensemble training complete. Artifacts saved to %s", out_dir)
    global _ml_last_train_time
    _ml_last_train_time = time.time()


# ================= SMART TRADING / SMC / CANDLE ENGINE =================
MAX_OPEN_TRADES = 100
MAX_SMART_SCAN_SYMBOLS = 40
SMART_UNIVERSE_ENABLED = True
SMART_SCAN_INTERVAL = 20
TRADE_COOLDOWN_SECONDS = 90
SIGNAL_SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal_screenshots")
os.makedirs(SIGNAL_SCREENSHOT_DIR, exist_ok=True)

BULL_CANDLE_NAMES = (
    "چکش","اینگالف صعودی","ستاره صبحگاهی","سه سرباز سفید","پیرسینگ",
    "انبرک کف","بولیش هارامی","کمربند نگه‌دارنده","Three Inside Up",
    "Three Outside Up","Rising Three Methods","Kicker صعودی",
    "Bullish Abandoned Baby","مارابوزو سبز"
)
BEAR_CANDLE_NAMES = (
    "مارابوزو قرمز","مرد به دار آویخته","ستاره دنباله‌دار","اینگالف نزولی",
    "ستاره عصرگاهی","ابر سیاه پوشاننده","سه کلاغ سیاه","هارامی نزولی",
    "انبرک سقف","Bearish Belt Hold","Three Inside Down","Three Outside Down",
    "Falling Three Methods","Advance Block","Bearish Stalled Pattern",
    "Bearish Abandoned Baby"
)

def detect_extended_candlestick_patterns(df):
    out = []
    try:
        if df is None or len(df) < 3:
            return out
        a,b,c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        def body(x): return abs(float(x["close"])-float(x["open"]))
        def rng(x): return max(float(x["high"])-float(x["low"]), 1e-12)
        cr=rng(c); cb=body(c); co=float(c["open"]); cc=float(c["close"])
        upper=float(c["high"])-max(co,cc); lower=min(co,cc)-float(c["low"])
        if cb/cr <= .10: out.append("دوجی")
        if cb/cr <= .25 and upper/cr>.25 and lower/cr>.25: out.append("فرفره")
        if lower >= 2*max(cb,1e-12) and upper <= cb*.8:
            out.append("چکش" if cc>=co else "مرد به دار آویخته")
        if upper >= 2*max(cb,1e-12) and lower <= cb*.8:
            out.append("ستاره دنباله‌دار")
        if cb/cr >= .82: out.append("مارابوزو سبز" if cc>co else "مارابوزو قرمز")
        ao,ac=float(a["open"]),float(a["close"]); bo,bc=float(b["open"]),float(b["close"])
        if ac<ao and bc>bo and bc>=ao and bo<=ac: out.append("اینگالف صعودی")
        if ac>ao and bc<bo and bc<=ao and bo>=ac: out.append("اینگالف نزولی")
        if abs(float(a["low"])-float(b["low"]))/max(float(a["low"]),1e-12)*100<.15 and bc>bo:
            out.append("انبرک کف")
        if abs(float(a["high"])-float(b["high"]))/max(float(a["high"]),1e-12)*100<.15 and bc<bo:
            out.append("انبرک سقف")
        if ac<ao and bc>bo and bo<=ac and bc<ao: out.append("بولیش هارامی")
        if ac>ao and bc<bo and bo>=ac and bc>ao: out.append("هارامی نزولی")
        if ac<ao and body(b)/rng(b)<.45 and bc>bo and bc>(ao+ac)/2: out.append("ستاره صبحگاهی")
        if ac>ao and body(b)/rng(b)<.45 and bc<bo and bc<(ao+ac)/2: out.append("ستاره عصرگاهی")
        if all(float(x["close"])>float(x["open"]) for x in (a,b,c)): out.append("سه سرباز سفید")
        if all(float(x["close"])<float(x["open"]) for x in (a,b,c)): out.append("سه کلاغ سیاه")
        ranges=(df["high"]-df["low"]).astype(float)
        if len(ranges)>=4 and float(ranges.iloc[-1])<=float(ranges.tail(4).min())+1e-15: out.append("NR4")
        if len(ranges)>=7 and float(ranges.iloc[-1])<=float(ranges.tail(7).min())+1e-15: out.append("NR7")
    except Exception:
        pass
    return list(dict.fromkeys(out))[-10:]

def detect_equal_high_low(df, tolerance_pct=.12):
    result={"eqh":[],"eql":[]}
    try:
        sh,sl=detect_swing_points(df,left=2,right=2)
        highs=[float(df["high"].iloc[i]) for i in sh[-10:]]
        lows=[float(df["low"].iloc[i]) for i in sl[-10:]]
        for vals,key in ((highs,"eqh"),(lows,"eql")):
            for i in range(len(vals)):
                for j in range(i+1,len(vals)):
                    mid=(vals[i]+vals[j])/2
                    if abs(vals[i]-vals[j])/max(mid,1e-12)*100 <= tolerance_pct:
                        result[key].append(mid)
        result["eqh"]=list(dict.fromkeys(result["eqh"]))[-4:]
        result["eql"]=list(dict.fromkeys(result["eql"]))[-4:]
    except Exception:
        pass
    return result

def detect_market_imbalance_score(df):
    try:
        bull,bear=detect_fvg(df)
        score=min(60,15*len(bull[-4:]))-min(60,15*len(bear[-4:]))
        if len(df)>=20:
            av=float(df["volume"].tail(20).mean()); v=float(df["volume"].iloc[-1])
            if av>0 and v/av>=1.4:
                score += 12 if float(df["close"].iloc[-1])>float(df["open"].iloc[-1]) else -12
        return float(max(-100,min(100,score))), bull[-3:], bear[-3:]
    except Exception:
        return 0.0,[],[]

def _v22_confidence_to_leverage(confidence: float) -> float:
    """Single source of truth: confirmed confidence determines leverage."""
    try: c = float(confidence)
    except Exception: c = 0.0
    if c >= 90: lev = 5.0
    elif c >= 85: lev = 4.0
    elif c >= 78: lev = 3.0
    elif c >= 72: lev = 2.0
    else: lev = 1.0
    # Respect an exchange/user hard ceiling if one exists.
    ceiling = safe_float(globals().get("MAX_MARGIN_LEVERAGE", 5.0), 5.0)
    return max(1.0, min(lev, ceiling))


def _v22_tp_profile(leverage: float):
    """Price-distance targets. Higher leverage tightens targets to limit account risk."""
    profiles = {
        1.0: (1.00, 2.00),
        2.0: (1.00, 2.00),
        3.0: (0.90, 1.80),
        4.0: (0.80, 1.60),
        5.0: (0.70, 1.40),
    }
    nearest = min(profiles, key=lambda x: abs(x-float(leverage)))
    return profiles[nearest]


def _v22_build_trade_plan(df, decision, entry, confidence=0.0):
    """Canonical TradePlan used by signal, live order, monitor and UI."""
    entry = float(entry or 0.0)
    if entry <= 0 or not str(decision).startswith(("BUY", "SELL")):
        return {"side": None, "entry": entry or None, "sl": None, "tp1": None,
                "tp2": None, "rr": 0.0, "leverage": 1.0, "confidence": float(confidence or 0)}
    lev = _v22_confidence_to_leverage(confidence)
    tp1_pct, tp2_pct = _v22_tp_profile(lev)
    atr = max(float(calculate_atr(df, 14) or 0.0), entry * 0.0025)
    sh, sl = detect_swing_points(df, left=2, right=2)
    low = float(df["low"].iloc[sl[-1]]) if sl else float(df["low"].tail(30).min())
    high = float(df["high"].iloc[sh[-1]]) if sh else float(df["high"].tail(30).max())
    if decision.startswith("BUY"):
        side = "long"
        # SL is structural/ATR based, but capped to a sane price-distance range.
        stop_dist = max(atr * 1.35, entry * 0.004)
        if low < entry:
            stop_dist = max(stop_dist, min(entry-low, entry*0.02))
        stop_dist = min(stop_dist, entry*0.025)
        sl_price = entry - stop_dist
        tp1 = entry * (1.0 + tp1_pct/100.0)
        tp2 = entry * (1.0 + tp2_pct/100.0)
    else:
        side = "short"
        stop_dist = max(atr * 1.35, entry * 0.004)
        if high > entry:
            stop_dist = max(stop_dist, min(high-entry, entry*0.02))
        stop_dist = min(stop_dist, entry*0.025)
        sl_price = entry + stop_dist
        tp1 = entry * (1.0 - tp1_pct/100.0)
        tp2 = entry * (1.0 - tp2_pct/100.0)
    rr = abs(tp2-entry) / max(abs(entry-sl_price), 1e-12)
    return {
        "side": side, "entry": entry, "sl": sl_price, "tp1": tp1, "tp2": tp2,
        "rr": rr, "leverage": lev, "confidence": float(confidence or 0.0),
        "tp1_price_pct": tp1_pct, "tp2_price_pct": tp2_pct,
        "tp1_leveraged_return_pct": tp1_pct*lev,
        "tp2_leveraged_return_pct": tp2_pct*lev,
    }


def build_smart_trade_plan(df, decision, entry, confidence=0.0):
    return _v22_build_trade_plan(df, decision, entry, confidence)

def get_smart_trade_universe(limit=MAX_SMART_SCAN_SYMBOLS):
    try:
        with state_lock:
            items=list(valid_symbols_map.items())
        scored=[]
        for sym,p in items:
            fp=safe_float(p)
            if fp<=0: continue
            bonus=4 if sym.endswith("USDT") else 2 if sym.endswith("IRT") else 0
            scored.append((bonus,fp,sym))
        scored.sort(reverse=True)
        return [x[2] for x in scored[:max(10,int(limit))]]
    except Exception:
        return list(SYMBOLS)[:limit]

def screenshot_current_signal(symbol, decision, entry, sl, tp1, tp2, score, confidence):
    try:
        from PIL import Image, ImageDraw, ImageFont
        path=os.path.join(SIGNAL_SCREENSHOT_DIR,f"{canonical(symbol)}_{int(time.time())}.png")
        img=Image.new("RGB",(1100,700),(15,23,42)); d=ImageDraw.Draw(img)
        try: font=ImageFont.truetype("arial.ttf",30)
        except Exception: font=ImageFont.load_default()
        lines=[
            "Nobitex AI Trader - Signal Snapshot",
            f"Symbol: {canonical(symbol)}",f"Decision: {decision}",
            f"Score: {score:.1f} | Confidence: {confidence:.1f}%",
            f"Entry: {format_price(entry)}",f"Stop Loss: {format_price(sl)}",
            f"Take Profit 1: {format_price(tp1)}",f"Take Profit 2: {format_price(tp2)}",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        y=40
        for i,line in enumerate(lines):
            d.text((50,y),line,fill=(226,232,240),font=font); y+=58
        img.save(path)
        return path
    except Exception:
        return None
# ================= END SMART ENGINE =================

# ---------------- Backtesting Engine ----------------
class VectorizedBacktester:
    def __init__(self, initial_capital: float=1000.0):
        self.initial_capital = initial_capital
        self.trades = []

    def run(self, df: pd.DataFrame, signals: List[Dict]) -> Dict:
        """
        Run vectorized backtest on historical signals.
        signals: list of dicts with entry_time, entry_price, side, predicted_move, stop_loss
        """
        try:
            if df.empty or not signals:
                return {"total_return": 0.0, "win_rate": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "trades": 0}

            equity = self.initial_capital
            equity_curve = [equity]
            wins = 0
            losses = 0

            for sig in signals:
                try:
                    entry_time = sig.get("time")
                    entry_price = sig.get("entry")
                    side = sig.get("side")
                    pred_move = sig.get("predicted_move_pct", 0.0)
                    stop_pct = sig.get("stop_loss_pct", STOP_LOSS_PCT_DEFAULT)

                    if not entry_time or not entry_price:
                        continue

                    # Find exit
                    entry_idx = df.index.get_loc(pd.Timestamp(entry_time))
                    if entry_idx < 0 or entry_idx >= len(df) - 1:
                        continue

                    future = df.iloc[entry_idx+1:entry_idx+25]  # ~6 hours on 15m
                    if future.empty:
                        continue

                    if side == "لانگ":
                        target = entry_price * (1 + pred_move / 100.0)
                        stop = entry_price * (1 - stop_pct / 100.0)
                        hit_target = future[future["high"] >= target]
                        hit_stop = future[future["low"] <= stop]
                        if not hit_target.empty and not hit_stop.empty:
                            if hit_target.index[0] < hit_stop.index[0]:
                                ret = pred_move
                                wins += 1
                            else:
                                ret = -stop_pct
                                losses += 1
                        elif not hit_target.empty:
                            ret = pred_move
                            wins += 1
                        elif not hit_stop.empty:
                            ret = -stop_pct
                            losses += 1
                        else:
                            final = (future["close"].iloc[-1] - entry_price) / entry_price * 100
                            ret = final
                            if final > 0:
                                wins += 1
                            else:
                                losses += 1
                    else:  # short
                        target = entry_price * (1 - pred_move / 100.0)
                        stop = entry_price * (1 + stop_pct / 100.0)
                        hit_target = future[future["low"] <= target]
                        hit_stop = future[future["high"] >= stop]
                        if not hit_target.empty and not hit_stop.empty:
                            if hit_target.index[0] < hit_stop.index[0]:
                                ret = pred_move
                                wins += 1
                            else:
                                ret = -stop_pct
                                losses += 1
                        elif not hit_target.empty:
                            ret = pred_move
                            wins += 1
                        elif not hit_stop.empty:
                            ret = -stop_pct
                            losses += 1
                        else:
                            final = (entry_price - future["close"].iloc[-1]) / entry_price * 100
                            ret = final
                            if final > 0:
                                wins += 1
                            else:
                                losses += 1

                    gross_ret_pct = float(ret)
                    cost_pct = backtest_round_trip_cost_pct()
                    # Costs are applied before leverage: fees/spread/slippage are
                    # execution drag on notional, while LEVERAGE magnifies P/L.
                    net_ret_pct = gross_ret_pct - cost_pct
                    leveraged_ret = net_ret_pct * LEVERAGE
                    equity *= (1 + leveraged_ret / 100.0)
                    equity_curve.append(equity)
                    self.trades.append({
                        "return": leveraged_ret,
                        "gross_return": gross_ret_pct * LEVERAGE,
                        "cost_pct": cost_pct,
                        "win": leveraged_ret > 0
                    })
                except Exception:
                    continue

            returns = [t["return"] for t in self.trades]
            total_ret = (equity - self.initial_capital) / self.initial_capital * 100
            # Win rate is based on NET return after configured execution costs.
            net_wins = sum(1 for t in self.trades if t.get("return",0) > 0)
            net_losses = sum(1 for t in self.trades if t.get("return",0) <= 0)
            win_rate = net_wins / (net_wins + net_losses) * 100 if (net_wins + net_losses) > 0 else 0
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * math.sqrt(252) if returns else 0

            # Max drawdown
            peak = self.initial_capital
            max_dd = 0.0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            return {
                "total_return": round(total_ret, 2),
                "win_rate": round(win_rate, 2),
                "sharpe": round(sharpe, 3),
                "max_drawdown": round(max_dd, 2),
                "trades": len(self.trades),
                "equity_curve": equity_curve
            }
        except Exception:
            return {"total_return": 0.0, "win_rate": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "trades": 0}

# ---------------- Talaye Evaluator (Enhanced) ----------------
def talaye_siah_eval_proxy(dfp: pd.DataFrame, dfc: pd.DataFrame, price: float, vol_usdt: float, sym: str):
    matched = []
    score_sum = 0.0
    weight_sum = 0.0
    if dfp is None or dfp.empty:
        return 0.0, []
    try:
        recent_bars = min(3, len(dfp))
        recent_vol = dfp["volume"].tail(recent_bars).sum()
        prev_len = min(24, max(0, len(dfp)-recent_bars))
        prev = dfp["volume"].iloc[-(recent_bars + prev_len):-recent_bars] if prev_len > 0 else pd.Series([])
        prev_avg = prev.mean() if not prev.empty else 0.0
        if prev_avg > 0:
            factor = recent_vol / max(prev_avg, 1e-9)
            w = 1.2
            weight_sum += w
            contrib = min(1.0, max(0.0, (factor - 1.0)/(TALAYE_VOL_FACTOR - 1.0)))
            if contrib > 0:
                matched.append({"name":"vol_breakout","contrib":contrib})
                score_sum += contrib * w
    except:
        pass
    try:
        rsi_series = compute_rsi_series(dfp, 14)
        rsi_vals = rsi_series.dropna().tolist()
        if rsi_vals:
            last_rsi = rsi_vals[-1]
            rising = False
            if len(rsi_vals) >= TALAYE_RSI_RISE_BARS:
                rising = all(rsi_vals[-TALAYE_RSI_RISE_BARS + i + 1] > rsi_vals[-TALAYE_RSI_RISE_BARS + i] for i in range(TALAYE_RSI_RISE_BARS - 1))
            if last_rsi < 40 and rising:
                w = 1.0
                weight_sum += w
                contrib = min(1.0, (40 - last_rsi)/20.0)
                matched.append({"name":"rsi_bounce","contrib":contrib})
                score_sum += contrib * w
    except:
        pass
    try:
        vwap = calculate_vwap_daily(dfp)
        if vwap:
            w = 0.8
            weight_sum += w
            if price > vwap:
                matched.append({"name":"vwap_above","contrib":1.0})
                score_sum += 1.0 * w
            elif price < vwap:
                matched.append({"name":"vwap_below","contrib":1.0})
                score_sum += 0.6 * w
    except:
        pass
    try:
        ub, ma, lb = calculate_bollinger_bands(dfp)
        if ub is not None:
            w = 0.9
            weight_sum += w
            pct = (price - ub)/ub if ub != 0 else 0
            if pct > 0:
                contrib = min(1.0, pct/0.02)
                matched.append({"name":"bb_breakout","contrib":contrib})
                score_sum += contrib * w
    except:
        pass
    try:
        delta = last_delta_cache.get(canonical(sym), 0.0)
        w = 1.1
        weight_sum += w
        if delta >= 0.25:
            matched.append({"name":"delta_buy","contrib":1.0})
            score_sum += 1.0 * w
        elif delta <= -0.25:
            matched.append({"name":"delta_sell","contrib":1.0})
            score_sum += 0.8 * w
    except:
        pass
    # NEW: Market structure bonus
    try:
        sh, sl = detect_swing_points(dfp)
        sweep_high, sweep_low = detect_liquidity_sweeps(dfp, sh, sl)
        if sweep_low and price > dfp["low"].iloc[sl[-1]] if sl else False:
            w = 1.3
            weight_sum += w
            matched.append({"name":"liquidity_sweep_bull","contrib":1.0})
            score_sum += 1.0 * w
        if sweep_high and price < dfp["high"].iloc[sh[-1]] if sh else False:
            w = 1.3
            weight_sum += w
            matched.append({"name":"liquidity_sweep_bear","contrib":1.0})
            score_sum += 1.0 * w
    except:
        pass
    norm = score_sum / (weight_sum if weight_sum > 0 else 1.0)
    return max(0.0, min(1.0, norm)), matched

# ---------------- Rules ----------------
RULES_META = [
    {"id":1,"rank":1,"name":"Orderflow Footprint Spike","hidden":"OF_FOOTPRINT_SPIKE","condition_text":"Candle vol > 5.5x avg AND delta > +0.75"},
    {"id":2,"rank":2,"name":"CVD Reset + Volume Explosion","hidden":"CVD_RESET_VOL_EXP","condition_text":"CVD resets ~0 AND next candle vol > 6.2x avg"},
    {"id":5,"rank":5,"name":"Liquidity Grab + 5x Volume","hidden":"LIQ_GRAB_5X","condition_text":"1h low below prev liquidity AND vol > 5x"},
    {"id":6,"rank":6,"name":"TPO Profile POC Shift","hidden":"TPO_POC_SHIFT","condition_text":"24h POC shift >=2 AND vol > 4.5x"},
    {"id":7,"rank":7,"name":"BOS Bullish + OB Support","hidden":"BOS_OB_SUPPORT","condition_text":"BOS bullish detected with bullish order block near price"},
    {"id":8,"rank":8,"name":"FVG Fill + Reversal","hidden":"FVG_REVERSAL","condition_text":"Price inside FVG with RSI divergence"},
    {"id":9,"rank":9,"name":"MTF Alignment Strong","hidden":"MTF_STRONG","condition_text":"MTF alignment >= 0.8 with volume confirmation"}
]
RULES_ENABLED = {r["id"]: True for r in RULES_META}

def eval_rule_1(sym, price, dfp, tf):
    try:
        vol_ratio = tf.get("1h", 0)
        delta = last_delta_cache.get(canonical(sym), 0.0)
        if vol_ratio >= 5.5 and delta >= 0.75:
            return True, {"vol_ratio":vol_ratio,"delta":delta}
    except:
        pass
    return False, {}

def eval_rule_2(sym, price, dfp, tf):
    try:
        vol_ratio = tf.get("1h", 0)
        dh = cache.get(f"delta_hist_{canonical(sym)}", [])
        last_sum = sum(dh[-12:]) if dh else 0.0
        if abs(last_sum) < 0.05 and vol_ratio >= 6.2:
            return True, {"cvd_sum":last_sum,"vol_ratio":vol_ratio}
    except:
        pass
    return False, {}

def eval_rule_5(sym, price, dfp, tf):
    try:
        vol_ratio = tf.get("1h", 0)
        if dfp.empty:
            return False, {}
        last_low = dfp["low"].iloc[-1]
        prev_lows = dfp["low"].iloc[-20:-1]
        if not prev_lows.empty and last_low < prev_lows.min() and vol_ratio >= 5.0:
            return True, {"last_low":last_low,"prev_min_low":prev_lows.min(),"vol_ratio":vol_ratio}
    except:
        pass
    return False, {}

def eval_rule_6(sym, price, dfp, tf):
    try:
        vol_ratio = tf.get("1h",0)
        if dfp.empty:
            return False, {}
        df24 = get_candles_cached(sym, 60, n=48, valid_symbols_map=valid_symbols_map)
        if df24.empty or len(df24) < 24:
            return False, {}
        last12 = df24["close"].tail(12)
        prev12 = df24["close"].iloc[-36:-24] if len(df24) >= 36 else df24["close"].head(12)
        mode_last = last12.mode().iloc[0] if not last12.mode().empty else last12.mean()
        mode_prev = prev12.mode().iloc[0] if not prev12.mode().empty else prev12.mean()
        shift = abs(mode_last - mode_prev)/max(mode_prev,1e-9)
        if shift >= 0.01 and vol_ratio >= 4.5:
            return True, {"shift_pct":shift,"vol_ratio":vol_ratio}
    except:
        pass
    return False, {}

def eval_rule_7(sym, price, dfp, tf):
    try:
        sh, sl = detect_swing_points(dfp)
        bullish_ob, _ = detect_order_blocks(dfp, sh, sl)
        if bullish_ob and any(abs(price - ob["low"])/price < 0.005 for ob in bullish_ob[-2:]):
            bos = detect_bos_choch(dfp, sh, sl)
            if bos["bos_bullish"]:
                return True, {"ob_near": True, "bos": True}
    except:
        pass
    return False, {}

def eval_rule_8(sym, price, dfp, tf):
    try:
        fvg_bull, fvg_bear = detect_fvg(dfp)
        if fvg_bull and any(price > f["bottom"] and price < f["top"] for f in fvg_bull[-2:]):
            rsi_series = compute_rsi_series(dfp, 14)
            bull_div, _ = detect_divergence(dfp["close"], rsi_series)
            if bull_div:
                return True, {"fvg_bull": True, "div": True}
    except:
        pass
    return False, {}

def eval_rule_9(sym, price, dfp, tf):
    try:
        mtf = calculate_mtf_alignment(sym)
        vol_ratio = tf.get("1h", 0)
        if mtf >= 0.8 and vol_ratio >= 2.0:
            return True, {"mtf": mtf, "vol_ratio": vol_ratio}
    except:
        pass
    return False, {}

RULE_EVALUATORS = {1:eval_rule_1, 2:eval_rule_2, 5:eval_rule_5, 6:eval_rule_6, 7:eval_rule_7, 8:eval_rule_8, 9:eval_rule_9}

def evaluate_rules_for_symbol(sym, price, dfp, tf):
    matched = []
    for meta in RULES_META:
        rid = meta["id"]
        if not RULES_ENABLED.get(rid, True):
            continue
        try:
            ok, details = RULE_EVALUATORS.get(rid, lambda *a, **k: (False, {}))(sym, price, dfp, tf)
            if ok:
                matched.append({"id":rid,"meta":meta,"details":details})
        except Exception:
            continue
    return matched

def load_weights():
    try:
        if os.path.exists("weights.json"):
            with open("weights.json", "r", encoding="utf-8") as f:
                w = json.load(f)
                for k in INDICATOR_WEIGHTS:
                    if k not in w:
                        w[k] = INDICATOR_WEIGHTS[k]
                return w
    except Exception:
        pass
    return INDICATOR_WEIGHTS.copy()


# ---------------- Trade History / Performance Analytics ----------------
TRADE_HISTORY_FIELDS = [
    "id","time","ts","symbol","side","direction","amount","usdt_value","margin_usdt",
    "position_value_usdt","leverage","entry_price","exit_price","exit_time","final_pct",
    "pnl_usdt","result","status","signal_score","signal_confidence","risk_pct"
]

def _calculate_trade_pnl_usdt(order: Dict[str, Any]) -> float:
    """Calculate realized P/L from executed quantity and actual entry/exit prices.

    The order quantity represents the full position notional. P/L is therefore
    quantity multiplied by the signed entry/exit price difference. This is
    valid for both current and legacy rows and prevents leverage being counted
    twice. Leverage changes margin and ROI percentage, not the position size.
    """
    amount = safe_float(order.get("amount"))
    entry = safe_float(order.get("entry_price"))
    exit_price = safe_float(order.get("exit_price"))
    if amount > 0 and entry > 0 and exit_price > 0:
        side = str(order.get("side") or "long").lower()
        direction_sign = -1.0 if side in ("short", "sell", "فروش") else 1.0
        return round(amount * (exit_price - entry) * direction_sign, 8)
    # Fallback for incomplete historical rows: usdt_value is notional,
    # while final_pct is leverage-adjusted return on margin.
    notional = safe_float(order.get("position_value_usdt")) or safe_float(order.get("usdt_value"))
    leverage = max(1.0, safe_float(order.get("leverage")) or 1.0)
    return round(notional * (safe_float(order.get("final_pct")) / leverage) / 100.0, 8)


def _trade_history_key(o: Dict[str, Any]) -> str:
    return str(o.get("id") or f"{o.get('symbol','')}|{o.get('ts','')}|{o.get('entry_price','')}")

def load_trade_history():
    global trade_history
    trade_history = []
    try:
        import csv
        if not os.path.exists(TRADE_HISTORY_FILE):
            return
        with open(TRADE_HISTORY_FILE, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    for k in ("ts","amount","usdt_value","margin_usdt","position_value_usdt","leverage","entry_price","exit_price","final_pct","pnl_usdt","signal_score","signal_confidence","risk_pct"):
                        if row.get(k) not in (None, ""):
                            row[k] = float(row[k])
                    trade_history.append(row)
                except Exception:
                    continue
    except Exception as e:
        logging.warning("trade history load failed: %s", e)

def save_closed_trade(order: Dict[str, Any]):
    if order.get("final_pct") is None or not order.get("exit_time"):
        return
    key = _trade_history_key(order)
    if any(_trade_history_key(x) == key for x in trade_history):
        return
    try:
        pnl = _calculate_trade_pnl_usdt(order)
        order["pnl_usdt"] = pnl
        order["id"] = order.get("id") or f"T{int(safe_float(order.get('ts'))*1000)}"
        row = {k: order.get(k, "") for k in TRADE_HISTORY_FIELDS}
        row["pnl_usdt"] = order.get("pnl_usdt", pnl)
        exists = os.path.exists(TRADE_HISTORY_FILE)
        import csv
        # Upgrade older CSV headers before appending the expanded accounting
        # schema; appending new-width rows under an old header corrupts columns.
        if exists and os.path.getsize(TRADE_HISTORY_FILE) > 0:
            with open(TRADE_HISTORY_FILE, "r", encoding="utf-8-sig", newline="") as src:
                reader = csv.DictReader(src)
                old_fields = list(reader.fieldnames or [])
                old_rows = list(reader) if old_fields != TRADE_HISTORY_FIELDS else None
            if old_rows is not None:
                tmp_path = TRADE_HISTORY_FILE + ".migrating"
                with open(tmp_path, "w", encoding="utf-8-sig", newline="") as dst:
                    migrated = csv.DictWriter(dst, fieldnames=TRADE_HISTORY_FIELDS)
                    migrated.writeheader()
                    for old_row in old_rows:
                        migrated.writerow({k: old_row.get(k, "") for k in TRADE_HISTORY_FIELDS})
                os.replace(tmp_path, TRADE_HISTORY_FILE)
        with open(TRADE_HISTORY_FILE, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRADE_HISTORY_FIELDS)
            if not exists or os.path.getsize(TRADE_HISTORY_FILE) == 0:
                w.writeheader()
            w.writerow(row)
        trade_history.append(dict(row))
    except Exception as e:
        logging.warning("save closed trade failed: %s", e)

def get_completed_trades():
    with state_lock:
        current = [dict(o) for o in auto_trade_log if o.get("final_pct") is not None and o.get("exit_time")]
    seen = set()
    out = []
    for o in trade_history + current:
        k = _trade_history_key(o)
        if k in seen:
            continue
        seen.add(k)
        try:
            # Always derive the displayed/statistical amount from the same
            # accounting rule used when the trade is persisted.
            o["pnl_usdt"] = _calculate_trade_pnl_usdt(o)
        except Exception:
            o["pnl_usdt"] = 0.0
        out.append(o)
    return out

def filter_trade_history(period="روزانه", symbol="همه", result="همه"):
    trades = get_completed_trades()
    now = datetime.now()
    start = None
    if period == "روزانه": start = now - timedelta(days=1)
    elif period == "هفتگی": start = now - timedelta(days=7)
    elif period == "ماهانه": start = now - timedelta(days=30)
    elif period == "سالانه": start = now - timedelta(days=365)
    for t in trades:
        dt = None
        try: dt = datetime.fromtimestamp(float(t.get("ts"))) if t.get("ts") else datetime.strptime(str(t.get("exit_time"))[:19], "%Y-%m-%d %H:%M:%S")
        except Exception: pass
        if start and (dt is None or dt < start): continue
        if symbol != "همه" and canonical(t.get("symbol")) != canonical(symbol): continue
        if result != "همه" and str(t.get("result")) != result: continue
        yield t

def summarize_trades(trades):
    trades=list(trades); wins=[t for t in trades if safe_float(t.get("pnl_usdt"))>0]; losses=[t for t in trades if safe_float(t.get("pnl_usdt"))<0]
    pnl=sum(safe_float(t.get("pnl_usdt")) for t in trades)
    pct=sum(safe_float(t.get("final_pct")) for t in trades)
    return {"trades":len(trades),"wins":len(wins),"losses":len(losses),"pnl":pnl,"pct":pct,
            "win_rate":(len(wins)/len(trades)*100 if trades else 0.0),
            "avg":(pnl/len(trades) if trades else 0.0),
            "best":max((safe_float(t.get("pnl_usdt")) for t in trades), default=0.0),
            "worst":min((safe_float(t.get("pnl_usdt")) for t in trades), default=0.0)}

# ---------------- Order Placement & Auto Trade ----------------
def _real_trading_is_enabled() -> bool:
    try:
        return bool(REAL_TRADING_ENABLED and REAL_TRADING_CONFIRMATION == REAL_TRADING_CONFIRM_PHRASE)
    except Exception:
        return False


def _extract_nobitex_order_id(result):
    """Best-effort extraction of an order id from different Nobitex response shapes."""
    try:
        if isinstance(result, dict):
            for key in ("order", "id", "orderId", "order_id"):
                value = result.get(key)
                if isinstance(value, dict):
                    nested = _extract_nobitex_order_id(value)
                    if nested:
                        return nested
                elif value not in (None, ""):
                    return str(value)
            for key in ("data", "result"):
                nested = _extract_nobitex_order_id(result.get(key))
                if nested:
                    return nested
    except Exception:
        pass
    return None


def _extract_nobitex_execution_price(result, fallback):
    """Best-effort extraction of a filled price; never trusts an invalid value."""
    try:
        candidates = []
        if isinstance(result, dict):
            candidates.extend([result.get("price"), result.get("avgPrice"), result.get("averagePrice")])
            order = result.get("order")
            if isinstance(order, dict):
                candidates.extend([order.get("price"), order.get("avgPrice"), order.get("averagePrice")])
        for value in candidates:
            price = safe_float(value)
            if price > 0:
                return price
    except Exception:
        pass
    return safe_float(fallback)


def _real_margin_symbol_allowed(market: str, direction: str = "long") -> tuple:
    """Return whether a symbol is actually enabled for Margin trading."""
    try:
        ok, settings, err = _real_margin_market_settings(canonical(market))
        if not ok:
            return False, err
        side = str(direction).lower()
        enabled = bool(settings.get("buyEnabled" if side == "long" else "sellEnabled", False))
        if not enabled:
            return False, f"بازار {canonical(market)} برای {side} در معاملات تعهدی فعال نیست."
        return True, ""
    except Exception as exc:
        return False, str(exc)

def _real_margin_market_settings(market: str):
    """Return live margin-market settings for a canonical symbol."""
    ok, data = nobitex_get_margin_markets()
    if not ok:
        return False, None, f"دریافت بازارهای تعهدی ناموفق: {data}"
    markets = data.get("markets", {}) if isinstance(data, dict) else {}
    target = canonical(market)
    settings = markets.get(target)
    if settings is None:
        # Be tolerant of API casing / alternate market keys.
        for k, v in markets.items():
            if canonical(str(k)) == target:
                settings = v
                break
    if not isinstance(settings, dict):
        return False, None, f"بازار {target} در معاملات تعهدی نوبیتکس فعال نیست."
    return True, settings, ""


def _real_get_wallet_balance(currency: str, wallets) -> float:
    """Extract a wallet balance from the several wallet response shapes Nobitex has used."""
    cur = str(currency).lower()
    best = 0.0
    def consume(obj):
        nonlocal best
        if not isinstance(obj, dict):
            return
        wc = str(obj.get("currency", obj.get("asset", ""))).lower()
        if wc == cur:
            for key in ("activeBalance", "availableBalance", "available", "balance"):
                if key in obj:
                    best = max(best, safe_float(obj.get(key)))
        for key in ("marginWallet", "margin_wallet", "wallet", "wallets"):
            child = obj.get(key)
            if isinstance(child, dict):
                consume(child)
            elif isinstance(child, list):
                for x in child:
                    consume(x)
    if isinstance(wallets, dict):
        consume(wallets)
        items = wallets.get("wallets", [])
        if isinstance(items, list):
            for x in items:
                consume(x)
        elif isinstance(items, dict):
            for k, x in items.items():
                if isinstance(x, dict):
                    y = dict(x)
                    y.setdefault("currency", k)
                    consume(y)
    elif isinstance(wallets, list):
        for x in wallets:
            consume(x)
    return best


def _real_get_conversion_price_to_rls(reference_symbol: str) -> float:
    """Get USDTIRT conversion price for IRT margin sizing.

    Conversion is a sizing operation, not an order execution operation.
    Therefore a fresh market/stats quote is an acceptable fallback when the
    USDTIRT orderbook endpoint is unavailable.  This avoids blocking every
    IRT trade just because the orderbook endpoint does not expose USDTIRT.
    """
    if not canonical(reference_symbol).endswith("IRT"):
        return 1.0

    # 1) Prefer the current market/stats mapping.
    try:
        p = safe_float(valid_symbols_map.get("USDTIRT"))
        if p > 0:
            return p
    except Exception:
        pass

    # 2) Cached quote, if still usable.
    try:
        cached = _ob_price_cache.get("USDTIRT")
        if cached and safe_float(cached[0]) > 0:
            return safe_float(cached[0])
    except Exception:
        pass

    # 3) Try a live two-sided quote only as a last resort.
    try:
        q = get_live_trade_quote("USDTIRT", "long")
        if q:
            for k in ("ask", "mid", "last", "bid"):
                v = safe_float(q.get(k))
                if v > 0:
                    return v
    except Exception:
        pass

    raise ValueError("قیمت USDTIRT برای تبدیل 10 USDT به RLS در دسترس نیست.")


def _real_submit_entry(sym: str, direction: str, amount: float, reference_price: float, leverage: float):
    """Open a real Nobitex margin position with correct IRT/USDT sizing."""
    if not _real_trading_is_enabled():
        return False, "معاملات واقعی فعال نیستند.", reference_price, None, None
    try:
        market = canonical(sym)
        base_cur, dst_cur = _nobitex_symbol_parts(market)
        side = "buy" if str(direction).lower() == "long" else "sell"

        ok_m, settings, err = _real_margin_market_settings(market)
        if not ok_m:
            return False, err, reference_price, None, None
        if side == "buy" and not bool(settings.get("buyEnabled", False)):
            return False, f"خرید تعهدی برای {market} فعال نیست.", reference_price, None, None
        if side == "sell" and not bool(settings.get("sellEnabled", False)):
            return False, f"فروش تعهدی برای {market} فعال نیست.", reference_price, None, None

        max_lev = max(1.0, safe_float(settings.get("maxLeverage"), 1.0))
        leverage = max(1.0, min(float(leverage), max_lev))

        # amount is calculated by place_order from the USDT notional. Recalculate
        # here as a final guard because IRT prices are denominated in RLS.
        position_value_usdt = _configured_trade_notional_usdt()
        if position_value_usdt <= 0:
            return False, "ارزش معامله AUTO_TRADE_USDT_VALUE نامعتبر است.", reference_price, None, None
        if dst_cur == "rls":
            usdt_irt = _real_get_conversion_price_to_rls(market)
            notional_dst = position_value_usdt * usdt_irt
        else:
            notional_dst = position_value_usdt
        amount = max(0.0, notional_dst / max(reference_price, 1e-18))
        # Respect API precision if market settings expose it.
        amount_precision = settings.get("amountPrecision", settings.get("amount_precision"))
        if amount_precision is not None:
            try:
                amount = round(amount, int(amount_precision))
            except Exception:
                pass
        if amount <= 0:
            return False, "مقدار سفارش پس از تبدیل ارز صفر شد.", reference_price, None, None

        ok_w, wallets = nobitex_get_wallets()
        if not ok_w:
            return False, f"دریافت کیف پول ناموفق: {wallets}", reference_price, None, None
        margin_balance = _real_get_wallet_balance(dst_cur, wallets)
        required_collateral = notional_dst / leverage
        wallet_use_pct = _configured_wallet_use_pct()
        allowed_collateral = margin_balance * (wallet_use_pct / 100.0)
        if margin_balance <= 0:
            return False, f"وجه تضمین کیف‌پول تعهدی {dst_cur} موجود نیست یا قابل تشخیص نیست.", reference_price, None, None
        if required_collateral > allowed_collateral * 1.001:
            logging.warning(
                "[MARGIN WALLET LIMIT] %s: required=%.8f %s, allowed=%.8f %s (wallet=%.8f, %.1f%%)",
                market, required_collateral, dst_cur, allowed_collateral, dst_cur,
                margin_balance, wallet_use_pct
            )
            return False, (
                f"سقف استفاده از کیف‌پول تعهدی رعایت نمی‌شود: "
                f"لازم {required_collateral:.8f} {dst_cur}، "
                f"حد مجاز {allowed_collateral:.8f} {dst_cur} "
                f"({wallet_use_pct:.1f}% از موجودی {margin_balance:.8f})"
            ), reference_price, None, None

        payload = {
            "srcCurrency": base_cur,
            "dstCurrency": dst_cur,
            "type": side,
            "execution": "market",
            "leverage": str(leverage),
            "amount": str(amount),
        }
        ok, result = _nobitex_request("POST", "/margin/orders/add", payload=payload)
        if not ok:
            logging.error("[REAL MARGIN ENTRY FAILED] %s %s amount=%s lev=%s: %s", side, market, amount, leverage, result)
            return False, result, reference_price, None, None

        # Nobitex can return HTTP 200 with status=failed; _nobitex_request should
        # already reject it, but keep this guard here too.
        if isinstance(result, dict) and str(result.get("status", "")).lower() in {"failed", "error", "rejected"}:
            return False, result, reference_price, None, None

        order_id = _extract_nobitex_order_id(result)
        fill_price = _extract_nobitex_execution_price(result, reference_price)
        position_id = None
        expected_side = "long" if side == "buy" else "short"
        for _ in range(12):
            okp, pres = nobitex_get_active_positions(market)
            if okp and isinstance(pres, dict):
                positions = pres.get("positions", []) or []
                candidates = []
                for x in positions:
                    if str(x.get("status", "")).lower() not in {"open", "active"}:
                        continue
                    ps = str(x.get("side", "")).lower()
                    if ps in {side, expected_side} or (side == "buy" and ps in {"long", "buy"}) or (side == "sell" and ps in {"short", "sell"}):
                        candidates.append(x)
                if candidates:
                    candidates.sort(key=lambda x: str(x.get("openedAt", x.get("createdAt", x.get("id", "")))), reverse=True)
                    pos = candidates[0]
                    position_id = pos.get("id")
                    fill_price = safe_float(pos.get("entryPrice"), fill_price)
                    amount = safe_float(pos.get("amount", pos.get("delegatedAmount", amount)), amount)
                    break
            time.sleep(0.5)
        if not position_id:
            logging.error("[REAL MARGIN ENTRY] order accepted but active position id was not found: %s", order_id)
            return False, "سفارش تعهدی پذیرفته شد ولی شناسه موقعیت باز پیدا نشد؛ برای جلوگیری از مدیریت اشتباه معامله، معامله محلی ساخته نشد.", fill_price, order_id, None
        logging.warning("[REAL MARGIN ENTRY] %s %s amount=%s leverage=%s price=%s order_id=%s position_id=%s", side.upper(), market, amount, leverage, fill_price, order_id, position_id)
        return True, result, fill_price, order_id, position_id
    except Exception as exc:
        logging.exception("[REAL MARGIN ENTRY EXCEPTION] %s", sym)
        return False, str(exc), reference_price, None, None


def _real_submit_exit(order_ref, reason: str):
    """Close a real Nobitex margin position using its position id."""
    if not _real_trading_is_enabled():
        return True
    try:
        if order_ref.get("real_exit_order_id"):
            return True
        position_id = order_ref.get("real_position_id")
        amount = safe_float(order_ref.get("margin_amount", order_ref.get("amount")))
        # Reconcile local state with the live position list before giving up.
        # This fixes legacy/local trades that survived without a saved position id.
        if not position_id:
            try:
                okp, pres = nobitex_get_active_positions(order_ref.get("symbol", ""))
                if okp and isinstance(pres, dict):
                    candidates = []
                    wanted = "long" if str(order_ref.get("side", "")).lower() == "long" else "short"
                    for pos in pres.get("positions", []) or []:
                        ps = str(pos.get("side", "")).lower()
                        if str(pos.get("status", "")).lower() in {"open", "active"} and ps in {wanted, "buy" if wanted == "long" else "sell"}:
                            candidates.append(pos)
                    if candidates:
                        candidates.sort(key=lambda x: str(x.get("openedAt", x.get("createdAt", x.get("id", "")))), reverse=True)
                        position_id = candidates[0].get("id")
                        amount = safe_float(candidates[0].get("amount", candidates[0].get("delegatedAmount", amount)), amount)
                        order_ref["real_position_id"] = position_id
                        order_ref["margin_amount"] = amount
            except Exception:
                logging.exception("[REAL MARGIN RECONCILE EXCEPTION] %s", order_ref.get("symbol"))
        if not position_id or amount <= 0:
            logging.warning("[REAL MARGIN EXIT SKIPPED] no live position found for %s; clearing stale local reference", order_ref.get("symbol"))
            order_ref["real_exit_order_id"] = "NO_LIVE_POSITION"
            order_ref["real_exit_reason"] = reason
            return True
        # Use the position-close endpoint. The opposite direction is handled
        # by Nobitex itself, which prevents accidentally opening a reverse spot trade.
        ok, result = nobitex_close_margin_position(position_id, amount, order_type="market")
        if not ok:
            logging.error("[REAL MARGIN EXIT FAILED] position=%s reason=%s: %s", position_id, reason, result)
            return False
        order_ref["real_exit_order_id"] = _extract_nobitex_order_id(result)
        order_ref["real_exit_reason"] = reason
        logging.warning("[REAL MARGIN EXIT] %s position=%s amount=%s reason=%s order_id=%s", order_ref.get("symbol"), position_id, amount, reason, order_ref.get("real_exit_order_id"))
        return True
    except Exception:
        logging.exception("[REAL MARGIN EXIT EXCEPTION] %s", order_ref.get("symbol"))
        return False


def _configured_trade_notional_usdt() -> float:
    try:
        if REAL_TRADING_SPOT_ONLY:
            return max(0.01, safe_float(SPOT_TRADE_USDT))
        return max(0.01, safe_float(MARGIN_TRADE_USDT))
    except Exception:
        return 5.0


def _configured_wallet_use_pct() -> float:
    try:
        if REAL_TRADING_SPOT_ONLY:
            return max(1.0, min(100.0, safe_float(SPOT_WALLET_USE_PCT)))
        return max(1.0, min(100.0, safe_float(MARGIN_WALLET_USE_PCT)))
    except Exception:
        return 75.0


def place_order(sym: str, direction: str, plan=None, score=0.0, confidence=0.0):
    key_sym = canonical(sym)
    with state_lock:
        opens=[o for o in auto_trade_log if o.get("result") is None]
        if len(opens) >= MAX_OPEN_TRADES:
            return None
        if any(o.get("symbol")==key_sym and o.get("result") is None for o in opens):
            return None
    # Trading entry must use a fresh executable quote: LONG buys at ASK, SHORT sells at BID.
    # The market/stats snapshot is intentionally not used as the primary entry price.
    price = get_executable_price(sym, direction)
    if price is None or price <= 0:
        logging.warning("place_order skipped for %s: fresh executable price unavailable", sym)
        return

    # NEW: Risk-adjusted position sizing using Kelly
    try:
        win_rate = 0.5
        avg_win = 2.0
        avg_loss = 1.0
        kelly = kelly_criterion(win_rate, avg_win, avg_loss)
        risk_pct = min(RISK_PER_TRADE_PCT, kelly * 100) if kelly > 0 else RISK_PER_TRADE_PCT
    except Exception:
        risk_pct = RISK_PER_TRADE_PCT

    # V22: leverage is determined exactly once from confirmed confidence.
    dynamic_leverage = _v22_confidence_to_leverage(confidence)
    if plan and plan.get("leverage"):
        dynamic_leverage = min(dynamic_leverage, safe_float(plan.get("leverage"), dynamic_leverage))

    # In REAL mode, never size/submit symbols that are not enabled on the
    # Nobitex Margin market list. This prevents the strategy from producing
    # repeated "market not active" aborts for ordinary spot-only symbols.
    if _real_trading_is_enabled():
        try:
            allowed, why = _real_margin_symbol_allowed(sym, direction)
            if not allowed:
                logging.info("[AUTO TRADE MARGIN MARKET FILTER] %s %s: %s", direction, sym, why)
                return None
        except Exception as exc:
            logging.warning("[AUTO TRADE MARGIN MARKET FILTER] %s: %s", sym, exc)
            return None

    # AUTO_TRADE_USDT_VALUE is the full position notional (same semantics as
    # the original UI). Leverage determines required margin, not quantity.
    # IRT markets use the independently configured fixed RLS amount.
    # 3,000,000 Toman = 30,000,000 RLS by default. USDT markets keep USDT sizing.
    is_irt = canonical(sym).endswith("IRT")
    if is_irt:
        try:
            rls_notional = MARGIN_TRADE_RLS if not REAL_TRADING_SPOT_ONLY else SPOT_TRADE_RLS
            rls_notional = max(1000.0, float(rls_notional))
            conversion = _real_get_conversion_price_to_rls(sym)
            notional_quote = rls_notional
            position_value_usdt = rls_notional / max(conversion, 1e-18)
        except Exception as exc:
            logging.error("[AUTO TRADE SIZING ABORTED] %s: %s", sym, exc)
            return None
    else:
        position_value_usdt = _configured_trade_notional_usdt()
        notional_quote = position_value_usdt
    margin_usdt = position_value_usdt / max(1.0, dynamic_leverage)
    amount = round(notional_quote / max(price, 1e-18), 8)
    if amount <= 0:
        logging.warning("place_order skipped for %s: calculated amount <= 0", sym)
        return None
    # Keep displayed USDT notional semantics; actual quote notional is tracked separately.
    margin_usdt = position_value_usdt / max(1.0, dynamic_leverage)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts = time.time()
    current_stop_loss_pct = get_stop_loss_pct()
    if plan and plan.get("sl"):
        auto_sl=float(plan["sl"]); auto_tp1=float(plan.get("tp1") or 0); auto_tp2=float(plan.get("tp2") or 0); auto_rr=float(plan.get("rr") or 0)
    elif direction=="long":
        fallback_plan=_v22_build_trade_plan(pd.DataFrame({"open":[price],"high":[price],"low":[price],"close":[price],"volume":[0]}), "BUY", price, confidence)
        auto_sl=float(fallback_plan.get("sl") or price*(1-current_stop_loss_pct/100)); auto_tp1=float(fallback_plan.get("tp1") or price*1.01); auto_tp2=float(fallback_plan.get("tp2") or price*1.02); auto_rr=float(fallback_plan.get("rr") or 0)
    else:
        fallback_plan=_v22_build_trade_plan(pd.DataFrame({"open":[price],"high":[price],"low":[price],"close":[price],"volume":[0]}), "SELL", price, confidence)
        auto_sl=float(fallback_plan.get("sl") or price*(1+current_stop_loss_pct/100)); auto_tp1=float(fallback_plan.get("tp1") or price*0.99); auto_tp2=float(fallback_plan.get("tp2") or price*0.98); auto_rr=float(fallback_plan.get("rr") or 0)
    if direction == "long":
        stop_loss_price = auto_sl
    else:
        stop_loss_price = auto_sl

    real_order_id = None
    real_position_id = None
    real_entry_price = price
    if _real_trading_is_enabled():
        ok_real, real_result, real_entry_price, real_order_id, real_position_id = _real_submit_entry(
            sym, direction, amount, price, dynamic_leverage
        )
        if not ok_real:
            logging.error("[REAL AUTO TRADE ABORTED] %s %s: %s", direction, sym, real_result)
            return None
        # Reprice the SAME canonical plan from the actual fill; no second TP/SL engine.
        price = real_entry_price if real_entry_price > 0 else price
        if plan and plan.get("tp1_price_pct"):
            p1, p2 = _v22_tp_profile(dynamic_leverage)
            if direction == "long":
                auto_tp1 = price * (1 + p1/100.0); auto_tp2 = price * (1 + p2/100.0)
            else:
                auto_tp1 = price * (1 - p1/100.0); auto_tp2 = price * (1 - p2/100.0)
            sl_gap = abs(float(plan.get("entry", price)) - float(plan.get("sl", price)))
            sl_gap = max(sl_gap, price*0.004)
            sl_gap = min(sl_gap, price*0.025)
            auto_sl = price-sl_gap if direction == "long" else price+sl_gap
            auto_rr = abs(auto_tp2-price)/max(sl_gap,1e-12)
        else:
            auto_sl = price * (1-current_stop_loss_pct/100) if direction == "long" else price*(1+current_stop_loss_pct/100)
            risk = abs(price-auto_sl)
            auto_tp1 = price + 1.5*risk if direction == "long" else price-1.5*risk
            auto_tp2 = price + 2.5*risk if direction == "long" else price-2.5*risk
            auto_rr = abs(auto_tp2-price)/max(risk,1e-12)
        stop_loss_price = auto_sl

    order = {
        "id": f"T{int(ts*1000)}",
        "real_mode": _real_trading_is_enabled(),
        "real_entry_order_id": real_order_id,
        "real_exit_order_id": None,
        "real_exit_reason": None,
        "real_position_id": real_position_id,
        "margin_amount": amount,
        "time": timestamp,
        "ts": ts,
        "symbol": key_sym,
        "direction": "خرید" if direction == "long" else "فروش",
        "side": direction,
        "amount": amount,
        # usdt_value is full position notional; margin_usdt is notional/leverage.
        # Quantity stays tied to notional so leverage is never counted twice.
        "usdt_value": round(position_value_usdt, 8),
        "margin_usdt": round(margin_usdt, 8),
        "position_value_usdt": round(position_value_usdt, 8),
        "leverage": dynamic_leverage,
        "entry_price": price,
        "current_price": price,
        "best_bid": price,
        "best_ask": price,
        "last_trade_price": price,
        "exit_price": None,
        "exit_time": None,
        "live_pct": 0.0,
        "final_pct": None,
        "status": "در جریان",
        "result": None,
        "peak_price": price,
        "trailing_stop": None,
        "trailing_active": False,
        "trailing_stage": 0,
        "trailing_locked_profit_pct": None,
        "trailing_retrace_pct": None,
        "stop_loss_price": stop_loss_price,
        "stop_loss_pct": current_stop_loss_pct,
        "take_profit_1": auto_tp1,
        "take_profit_2": auto_tp2,
        "tp1_hit": False,
        "rr_plan": auto_rr,
        "signal_score": float(score),
        "signal_confidence": float(confidence),
        "slippage": False,
        "risk_pct": risk_pct,
        # Compact live path used by the Trade Monitor chart.
        "price_history": [{"ts": ts, "price": price}],
        "monitor_priority": 0,
    }
    with state_lock:
        auto_trade_log.append(order)
    if _real_trading_is_enabled():
        logging.warning("[REAL MARGIN AUTO TRADE] %s %s @ %s | leverage=%s | local SL/TP monitor | order_id=%s position_id=%s", direction.upper(), key_sym, price, dynamic_leverage, real_order_id, real_position_id)
    else:
        logging.info("[SIM AUTO TRADE] %s %s @ %s | SL: %s (%s%%) | Risk: %.2f%%", direction.upper(), key_sym, price, stop_loss_price, current_stop_loss_pct, risk_pct)
    try:
        if app:
            app.root.after(100, app.update_auto_trade_tab)
    except Exception:
        pass


def get_live_trade_price(symbol, side: Optional[str]=None):
    """Return a fresh, side-aware executable price for an open trade.

    LONG exits are valued on BID; SHORT exits are valued on ASK. If the
    orderbook has no two-sided quote, the helper falls back to last/mid.
    """
    return get_executable_price(symbol, side)

def _v18_log_trade_closed(order_ref, reason):
    try:
        _v17_log('TRADE_CLOSED',
                 trade_id=order_ref.get('id'),
                 symbol=order_ref.get('symbol'),
                 side=order_ref.get('side'),
                 reason=reason,
                 entry_price=_v17_num(order_ref.get('entry_price')),
                 exit_price=_v17_num(order_ref.get('exit_price')),
                 stop_loss=_v17_num(order_ref.get('stop_loss_price')),
                 tp1=_v17_num(order_ref.get('take_profit_1')),
                 tp2=_v17_num(order_ref.get('take_profit_2')),
                 leverage=_v17_num(order_ref.get('leverage'), 1),
                 final_pct=_v17_num(order_ref.get('final_pct')),
                 duration_sec=round(max(0.0, time.time()-_v17_num(order_ref.get('ts'), time.time())),2))
    except Exception as exc:
        logging.exception('V18 close telemetry failed: %s', exc)


def evaluate_auto_trades():
    SLIPPAGE_THRESHOLD_PCT = 5.0
    while True:
        try:
            with state_lock:
                local_auto_trade_copy = list(auto_trade_log)
            if not local_auto_trade_copy:
                time.sleep(3)
                continue
            now = time.time()
            now_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
            trailing_percent = get_trailing_percent()
            trailing_activate = get_trailing_activate()

            for order in local_auto_trade_copy:
                try:
                    if order.get("result") is not None:
                        continue
                    current = get_live_trade_price(order["symbol"], order.get("side"))
                    if current and current > 0:
                        q = get_live_trade_quote(order["symbol"], order.get("side")) or {}
                        with state_lock:
                            for o in auto_trade_log:
                                if o.get("ts") == order.get("ts") and o.get("symbol") == order.get("symbol"):
                                    o["current_price"] = current
                                    o["best_bid"] = safe_float(q.get("bid"))
                                    o["best_ask"] = safe_float(q.get("ask"))
                                    o["last_trade_price"] = safe_float(q.get("last"))
                                    hist = o.setdefault("price_history", [])
                                    hnow = time.time()
                                    # Avoid excessive points when the same price is returned.
                                    if not hist or abs(safe_float(hist[-1].get("price")) - current) > 0:
                                        hist.append({"ts": hnow, "price": current})
                                    # Keep the chart lightweight.
                                    if len(hist) > 180:
                                        del hist[:-180]
                                    order_ref = o
                                    break
                            else:
                                order_ref = None
                        if order_ref is None:
                            continue
                    else:
                        order_ref = order

                    base_pct = (order_ref["current_price"] - order_ref["entry_price"]) / order_ref["entry_price"] * 100
                    order_leverage = max(1.0, safe_float(order_ref.get("leverage")) or 1.0)
                    leveraged_pct = base_pct * order_leverage if order_ref["side"] == "long" else -base_pct * order_leverage
                    with state_lock:
                        order_ref["live_pct"] = round(leveraged_pct, 4)
                        qty_for_live = max(0.0, safe_float(order_ref.get("amount")))
                        side_sign = -1.0 if order_ref.get("side") == "short" else 1.0
                        order_ref["live_pnl_usdt"] = round(qty_for_live * (safe_float(order_ref.get("current_price")) - safe_float(order_ref.get("entry_price"))) * side_sign, 8)

                    tp1=safe_float(order_ref.get("take_profit_1")); tp2=safe_float(order_ref.get("take_profit_2"))
                    if tp1>0 and not order_ref.get("tp1_hit"):
                        hit=(order_ref["side"]=="long" and order_ref["current_price"]>=tp1) or (order_ref["side"]=="short" and order_ref["current_price"]<=tp1)
                        if hit:
                            with state_lock:
                                order_ref["tp1_hit"]=True
                                order_ref["status"]="TP1 فعال / انتقال SL به نقطه ورود"
                                order_ref["stop_loss_price"]=order_ref["entry_price"]
                    if tp2>0:
                        hit=(order_ref["side"]=="long" and order_ref["current_price"]>=tp2) or (order_ref["side"]=="short" and order_ref["current_price"]<=tp2)
                        if hit:
                            if not _real_submit_exit(order_ref, "TP2"):
                                continue
                            with state_lock:
                                order_ref["exit_price"]=order_ref["current_price"]
                                order_ref["exit_time"]=now_str
                                bp=(order_ref["exit_price"]-order_ref["entry_price"])/order_ref["entry_price"]*100
                                order_ref["final_pct"]=round(bp*order_leverage if order_ref["side"]=="long" else -bp*order_leverage,4)
                                order_ref["result"]="TP2"
                                order_ref["status"]="بسته (TP2)"
                            order_ref["pnl_usdt"] = _calculate_trade_pnl_usdt(order_ref)
                            _v18_log_trade_closed(order_ref, "TP2")
                            # TP2 is terminal for this trade. Do not let the same
                            # price sample fall through into SL/trailing logic.
                            continue

                    if not order_ref["trailing_active"]:
                        stop_hit = (
                            (order_ref["side"] == "long" and order_ref["current_price"] <= order_ref["stop_loss_price"]) or
                            (order_ref["side"] == "short" and order_ref["current_price"] >= order_ref["stop_loss_price"])
                        )
                        if stop_hit:
                            if not _real_submit_exit(order_ref, "STOP_LOSS"):
                                continue
                            with state_lock:
                                order_ref["exit_price"] = order_ref["current_price"]
                                order_ref["exit_time"] = now_str
                                base_pct_exit = (order_ref["exit_price"] - order_ref["entry_price"]) / order_ref["entry_price"] * 100
                                leveraged_final = base_pct_exit * order_leverage if order_ref["side"] == "long" else -base_pct_exit * order_leverage
                                order_ref["final_pct"] = round(leveraged_final, 4)
                                order_ref["result"] = "سپر دفاعی"
                                order_ref["status"] = "بسته (سپر دفاعی)"
                                expected_final_pct = - (order_ref.get("stop_loss_pct", STOP_LOSS_PCT_DEFAULT) / 100.0) * order_leverage
                                diff = abs(order_ref["final_pct"] - expected_final_pct)
                            order_ref["pnl_usdt"] = _calculate_trade_pnl_usdt(order_ref)
                            _v18_log_trade_closed(order_ref, "STOP_LOSS")
                            if diff >= SLIPPAGE_THRESHOLD_PCT:
                                with state_lock:
                                    order_ref["slippage"] = True
                                log_row = {
                                    "time": now_str,
                                    "symbol": order_ref["symbol"],
                                    "entry": order_ref["entry_price"],
                                    "exit": order_ref["exit_price"],
                                    "expected_final_pct": expected_final_pct,
                                    "actual_final_pct": order_ref["final_pct"],
                                    "stop_loss_pct": order_ref.get("stop_loss_pct"),
                                    "diff": diff
                                }
                                try:
                                    with open(SLIPPAGE_LOG, "a", encoding="utf-8") as f:
                                        f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
                                except Exception:
                                    pass
                                logging.warning("SLIPPAGE detected for %s: expected=%.3f actual=%.3f", order_ref["symbol"], expected_final_pct, order_ref["final_pct"])
                            else:
                                with state_lock:
                                    order_ref["slippage"] = False
                            say_farsi(f"سپر دفاعی فعال شد برای {order_ref['symbol']}", priority=True)
                            continue

                    if order_ref["side"] == "long":
                        with state_lock:
                            if order_ref["current_price"] > order_ref["peak_price"]:
                                order_ref["peak_price"] = order_ref["current_price"]
                    else:
                        with state_lock:
                            if order_ref["current_price"] < order_ref["peak_price"]:
                                order_ref["peak_price"] = order_ref["current_price"]

                    # ============================================================
                    # USER TRAILING POLICY
                    # Stage 1: activate at +1% raw price profit, allow 70%
                    # retracement of the achieved profit (keep 30%).
                    # Stage 2: once profit reaches +2%, allow only 50%
                    # retracement (keep 50%).
                    # Trailing is NEVER allowed to close a trade below entry.
                    # Defensive Shield remains responsible for controlled losses.
                    # ============================================================
                    entry_price = max(1e-12, safe_float(order_ref.get("entry_price")))
                    current_price = safe_float(order_ref.get("current_price"))
                    side = str(order_ref.get("side", "long"))
                    raw_profit_pct = ((current_price-entry_price)/entry_price*100.0) if side == "long" else ((entry_price-current_price)/entry_price*100.0)
                    peak_price = safe_float(order_ref.get("peak_price"))
                    peak_profit_pct = ((peak_price-entry_price)/entry_price*100.0) if side == "long" else ((entry_price-peak_price)/entry_price*100.0)
                    activate_pct = max(0.01, get_trailing_activate())
                    stage2_pct = max(activate_pct, get_trailing_stage2())
                    s1_retrace = max(0.0, min(95.0, get_trailing_stage1_retrace()))
                    s2_retrace = max(0.0, min(95.0, get_trailing_stage2_retrace()))

                    if not order_ref["trailing_active"] and raw_profit_pct >= activate_pct and is_trailing_enabled():
                        with state_lock:
                            order_ref["trailing_active"] = True
                            order_ref["trailing_stage"] = 2 if raw_profit_pct >= stage2_pct else 1
                            order_ref["trailing_retrace_pct"] = s2_retrace if raw_profit_pct >= stage2_pct else s1_retrace
                            keep_fraction = 1.0 - order_ref["trailing_retrace_pct"] / 100.0
                            locked_profit = max(0.0, peak_profit_pct * keep_fraction)
                            order_ref["trailing_locked_profit_pct"] = locked_profit
                            if side == "long":
                                order_ref["trailing_stop"] = entry_price * (1.0 + locked_profit/100.0)
                            else:
                                order_ref["trailing_stop"] = entry_price * (1.0 - locked_profit/100.0)
                        logging.info("[TRAILING] %s activated | profit=%.3f%% peak=%.3f%% stage=%s retrace=%.1f%% lock=%.3f%%", order_ref.get("symbol"), raw_profit_pct, peak_profit_pct, order_ref.get("trailing_stage"), order_ref.get("trailing_retrace_pct"), order_ref.get("trailing_locked_profit_pct"))

                    if order_ref["trailing_active"]:
                        # Re-evaluate stage from the BEST profit reached. Once
                        # stage 2 is reached, never downgrade back to stage 1.
                        stage = 2 if peak_profit_pct >= stage2_pct else 1
                        retrace = s2_retrace if stage == 2 else s1_retrace
                        keep_fraction = 1.0 - retrace / 100.0
                        locked_profit = max(0.0, peak_profit_pct * keep_fraction)
                        # Hard floor: trailing may not create a negative exit.
                        locked_profit = max(0.0, locked_profit)
                        if stage >= int(order_ref.get("trailing_stage") or 1):
                            with state_lock:
                                order_ref["trailing_stage"] = stage
                                order_ref["trailing_retrace_pct"] = retrace
                                order_ref["trailing_locked_profit_pct"] = locked_profit
                                if side == "long":
                                    new_stop = entry_price * (1.0 + locked_profit/100.0)
                                    if order_ref.get("trailing_stop") is None or new_stop > safe_float(order_ref.get("trailing_stop")):
                                        order_ref["trailing_stop"] = new_stop
                                else:
                                    new_stop = entry_price * (1.0 - locked_profit/100.0)
                                    if order_ref.get("trailing_stop") is None or new_stop < safe_float(order_ref.get("trailing_stop")):
                                        order_ref["trailing_stop"] = new_stop

                        trail_stop = safe_float(order_ref.get("trailing_stop"))
                        hit = (side == "long" and current_price <= trail_stop) or (side != "long" and current_price >= trail_stop)
                        # Never close via trailing at/below a loss. A tick below
                        # entry is ignored here; the shield handles loss exits.
                        if hit and raw_profit_pct >= 0.0 and trail_stop > 0:
                            if not _real_submit_exit(order_ref, "TRAILING_PROFIT_LOCK"):
                                continue
                            with state_lock:
                                order_ref["exit_price"] = current_price
                                order_ref["exit_time"] = now_str
                                base_pct_exit = raw_profit_pct
                                order_ref["final_pct"] = round(base_pct_exit * order_leverage, 4)
                                order_ref["result"] = "تریلینگ"
                                order_ref["status"] = "بسته (Trailing / Profit Lock)"
                            order_ref["pnl_usdt"] = _calculate_trade_pnl_usdt(order_ref)
                            _v18_log_trade_closed(order_ref, "TRAILING_PROFIT_LOCK")
                            continue

                    if order_ref["result"] is None and now - order_ref["ts"] >= EVAL_SECONDS:
                        if not _real_submit_exit(order_ref, "TIMEOUT"):
                            continue
                        with state_lock:
                            order_ref["final_pct"] = round(leveraged_pct, 4)
                            order_ref["exit_price"] = order_ref["current_price"]
                            order_ref["exit_time"] = now_str
                            order_ref["result"] = "درست" if leveraged_pct > 0 else "غلط"
                        order_ref["pnl_usdt"] = _calculate_trade_pnl_usdt(order_ref)
                        _v18_log_trade_closed(order_ref, "TIMEOUT_PROFIT" if leveraged_pct > 0 else "TIMEOUT_LOSS")
                except Exception as exc:
                    logging.exception("Auto-trade evaluation failed for %s: %s", order.get("symbol"), exc)
                    continue

            # Persist newly completed trades for daily/weekly/monthly/yearly analytics.
            try:
                with state_lock:
                    closed_now = [o for o in auto_trade_log if o.get("final_pct") is not None and o.get("exit_time")]
                for closed in closed_now:
                    save_closed_trade(closed)
            except Exception:
                logging.exception("trade history persistence failed")

            # Update equity curve for risk management
            try:
                with state_lock:
                    total_equity = 1000.0
                    for o in auto_trade_log:
                        if o.get("final_pct") is not None:
                            total_equity *= (1 + o["final_pct"] / 100.0)
                    _equity_curve.append(total_equity)
            except Exception:
                pass

            try:
                if app:
                    app.root.after(0, app.update_auto_trade_tab)
                    app.root.after(0, app.update_risk_tab)
            except Exception:
                pass
            time.sleep(4)
        except Exception:
            time.sleep(4)

# ---------------- Utility functions ----------------
def build_price_map(symbols: List[str]) -> Dict[str, float]:
    pm = {}
    for s in symbols:
        try:
            p = get_price_from_orderbook(s, valid_symbols_map)
            if p:
                pm[canonical(s)] = p
        except Exception:
            continue
    return pm

def ensure_csv_header():
    if not os.path.exists("signal_history.csv"):
        with open("signal_history.csv","w",encoding="utf-8") as f:
            f.write("time,symbol,type,entry,result,profit,exit_time,expire_time\n")
    if not os.path.exists("signal_accuracy.csv"):
        with open("signal_accuracy.csv","w",encoding="utf-8") as f:
            f.write("time,symbol,direction,entry,check_price,result,pct_change\n")

def check_pending_signals():
    while True:
        try:
            with state_lock:
                now = time.time()
                for rec in list(signal_log):
                    if rec.get("expired") or rec.get("result"):
                        continue
                    _eff_target = rec.get("manual_target_price") or rec.get("target_price")
                    _eff_entry = rec.get("manual_entry") or rec.get("entry")
                    if _eff_target and _eff_entry:
                        cur = get_price_from_orderbook(rec["symbol"], valid_symbols_map)
                        if cur is None:
                            continue
                        rec["current_price"] = cur
                        if rec["side"] == "لانگ":
                            if cur >= _eff_target:
                                rec["exit"] = cur
                                rec["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                rec["realized_pct"] = (cur - _eff_entry) / _eff_entry * 100
                                rec["result"] = "درست"
                        else:
                            if cur <= _eff_target:
                                rec["exit"] = cur
                                rec["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                rec["realized_pct"] = (_eff_entry - cur) / _eff_entry * 100
                                rec["result"] = "درست"
        except Exception:
            pass
        time.sleep(8)

def expire_logged_signals_worker():
    while True:
        try:
            with state_lock:
                now = time.time()
                for rec in signal_log:
                    if rec.get("expired") or rec.get("result"):
                        continue
                    if rec.get("time") and now - rec.get("time") > EXPIRE_SECONDS:
                        rec["expired"] = True
                        rec["expire_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        try:
                            if not rec.get("result"):
                                lp = rec.get("live_pct")
                                if lp is None:
                                    curp = rec.get("current_price")
                                    ep = rec.get("manual_entry") or rec.get("entry")
                                    if curp and ep:
                                        if rec.get("side") == "لانگ":
                                            lp = (curp - ep) / ep * 100.0
                                        else:
                                            lp = (ep - curp) / ep * 100.0
                                if lp is not None:
                                    rec["result"] = "درست" if lp > 0 else "غلط"
                                    rec["realized_pct"] = round(lp, 4)
                        except Exception:
                            pass
        except Exception:
            pass
        time.sleep(60)

def treeview_sort_column(tv: ttk.Treeview, col: str, reverse: bool=False):
    try:
        l = [(tv.set(k, col), k) for k in tv.get_children("")]
        try:
            l.sort(key=lambda t: float(re.sub(r"[^\d\.\-]", "", t[0])) if t[0] not in ("", None) else float("-inf"), reverse=reverse)
        except Exception:
            l.sort(key=lambda t: t[0], reverse=reverse)
        for index, (_, k) in enumerate(l):
            tv.move(k, "", index)
    except Exception:
        pass

def prepare_symbol_mapping():
    global valid_symbols_map, canonical_to_api, SYMBOLS
    logging.info("prepare_symbol_mapping: fetching market/stats...")
    api_map = fetch_market_stats()
    if not api_map:
        logging.warning("market/stats fetch failed — using fallback")
        with state_lock:
            valid_symbols_map = {canonical(s): None for s in SYMBOLS}
        return
    canonical_to_api = {}
    valid_symbols_map = {}
    for api_sym, price in api_map.items():
        can = canonical(api_sym)
        canonical_to_api[can] = api_sym
        valid_symbols_map[can] = price
    if SMART_UNIVERSE_ENABLED:
        mapped = [canonical(x) for x in canonical_to_api.keys() if canonical(x)]
        with state_lock:
            SYMBOLS = mapped
            valid_symbols_map = {k: valid_symbols_map.get(k) for k in mapped if k in valid_symbols_map}
        logging.info("Smart universe enabled: %d exchange symbols available", len(SYMBOLS))
    else:
        mapped = []
        for u in USER_SYMBOLS:
            nu = canonical(u)
            if nu in canonical_to_api:
                mapped.append(nu)
        if mapped:
            with state_lock:
                SYMBOLS = mapped
                valid_symbols_map = {k: valid_symbols_map.get(k) for k in mapped if k in valid_symbols_map}
            logging.info("Mapped %d symbols", len(mapped))
        else:
            with state_lock:
                SYMBOLS = [canonical(s) for s in USER_SYMBOLS]

# ---------------- Alerts ----------------
def trigger_talaye_alert(sym: str, price: float, score: float):
    now = time.time()
    last = talaye_alerts.get(canonical(sym), 0)
    if now - last < 600:
        return
    talaye_alerts[canonical(sym)] = now
    try:
        import winsound
        winsound.Beep(2000,180); time.sleep(0.05)
        winsound.Beep(2300,200); time.sleep(0.05)
        winsound.Beep(2600,320)
    except Exception:
        pass
    say_farsi(f"طلایه سیاه: {sym} امتیاز {int(round(score))}", priority=True)
    send_telegram(f"طلایه سیاه: {sym} entry={price} score={score}")

def trigger_normal_alert(sym: str, price: float, score: float):
    try:
        import winsound
        winsound.Beep(1200,120); time.sleep(0.03); winsound.Beep(1350,120)
    except Exception:
        pass
    try:
        say_farsi(f"سیگنال معمولی: {sym} امتیاز {int(round(score))}")
    except:
        pass
    send_telegram(f"Signal: {sym} entry={price} score={score}")

# ---------------- Main UI Class (Enhanced) ----------------
class NobitexScalperPro:
    def __init__(self, root: tk.Tk):
        global app
        app = self
        self.root = root
        self.root.title("Nobitex AI Scalper Pro V20 — Profit-Lock Trailing + Defensive Shield")
        self.root.geometry("1800x1080")
        self.running = False

        self.auto_trade = tk.BooleanVar(value=False)
        self.real_trading_enabled = tk.BooleanVar(value=False)
        self.real_trading_confirm = tk.StringVar(value="")
        self.tts_enabled = tk.BooleanVar(value=bool(TTS_AVAILABLE))
        self.talaye_enabled = tk.BooleanVar(value=TALAYE_ENABLED)
        self.trailing_enabled = tk.BooleanVar(value=True)
        self.ml_enabled = tk.BooleanVar(value=True)
        self._ml_enabled_state = True
        self._auto_trade_state = False
        self.show_structure = tk.BooleanVar(value=True)

        self.margin_wallet_use_pct_var = tk.DoubleVar(value=float(_RISK_SETTINGS.get("margin_wallet_use_pct", MARGIN_WALLET_USE_PCT)))
        self.margin_trade_usdt_var = tk.DoubleVar(value=float(_RISK_SETTINGS.get("margin_trade_usdt", MARGIN_TRADE_USDT)))
        self.spot_wallet_use_pct_var = tk.DoubleVar(value=float(_RISK_SETTINGS.get("spot_wallet_use_pct", SPOT_WALLET_USE_PCT)))
        self.spot_trade_usdt_var = tk.DoubleVar(value=float(_RISK_SETTINGS.get("spot_trade_usdt", SPOT_TRADE_USDT)))
        self.margin_trade_rls_var = tk.DoubleVar(value=float(_RISK_SETTINGS.get("margin_trade_rls", MARGIN_TRADE_RLS)))
        self.spot_trade_rls_var = tk.DoubleVar(value=float(_RISK_SETTINGS.get("spot_trade_rls", SPOT_TRADE_RLS)))

        self.trailing_activate_var = tk.DoubleVar(value=float(_RISK_SETTINGS.get("trailing_activate_pct", TRAILING_ACTIVATE_PCT_DEFAULT)))
        self.trailing_stage2_var = tk.DoubleVar(value=float(_RISK_SETTINGS.get("trailing_stage2_pct", TRAILING_STAGE2_PCT_DEFAULT)))
        self.trailing_stage1_retrace_var = tk.DoubleVar(value=float(_RISK_SETTINGS.get("trailing_stage1_retrace_pct", TRAILING_STAGE1_RETRACE_DEFAULT)))
        self.trailing_stage2_retrace_var = tk.DoubleVar(value=float(_RISK_SETTINGS.get("trailing_stage2_retrace_pct", TRAILING_STAGE2_RETRACE_DEFAULT)))
        self.trailing_percent_var = self.trailing_stage2_retrace_var  # compatibility alias
        self.stop_loss_pct_var = tk.DoubleVar(value=float(_RISK_SETTINGS.get("defensive_shield_pct", STOP_LOSS_PCT_DEFAULT)))
        self.trailing_enabled.set(bool(_RISK_SETTINGS.get("trailing_enabled", True)))
        for _v in (self.trailing_activate_var, self.trailing_stage2_var, self.trailing_stage1_retrace_var, self.trailing_stage2_retrace_var, self.stop_loss_pct_var):
            try:
                _v.trace_add("write", lambda *_a: _save_settings())
            except Exception:
                pass

        self.data = {canonical(s): {"prev_price": None} for s in SYMBOLS}
        self.executor = ThreadPoolExecutor(max_workers=max(8, executor_workers))
        self.talaye_blink_symbols = set()
        self.blink_state = False
        self.sort_state = {"tree": None, "col": None, "reverse": False}
        self._scale_blocks: Dict[str, Any] = {}
        load_trade_history()

        self._build_ui()
        # Start on the dedicated live Trade Monitor so the active positions
        # are immediately in front of the operator.
        try:
            self.notebook.select(self.trade_monitor_tab)
        except Exception:
            pass
        self._sync_threadsafe_flags()
        # Trailing is enabled by default for simulated auto-trade; users can
        # explicitly switch it off from the checkbox.
        try:
            if not bool(self.trailing_enabled.get()):
                self.trailing_enabled.set(True)
        except Exception: pass
        self.on_trailing_toggle_initial()
        if TTS_AVAILABLE:
            init_tts()
        self.root.after(700, self._blink)
        threading.Thread(target=check_pending_signals, daemon=True).start()
        threading.Thread(target=expire_logged_signals_worker, daemon=True).start()
        threading.Thread(target=evaluate_auto_trades, daemon=True).start()
        logging.info("Live trade monitor enabled: price/PnL refresh loop active")
        # FIX 9: Never block Tkinter while building the 500+ symbol mapping.
        # The old synchronous call made the window appear frozen during startup.
        def _map_worker():
            try:
                self.set_tab_activity("نمای کلی", "در حال آماده‌سازی بازار", "active")
                prepare_symbol_mapping()
                self.root.after(0, lambda: self.set_tab_activity("نمای کلی", "بازار آماده", "done"))
                self.root.after(0, self.refresh_overview_prices)
            except Exception as e:
                logging.exception("background symbol mapping failed: %s", e)
                self.root.after(0, lambda err=str(e): self.set_tab_activity("نمای کلی", f"خطا: {err}", "done"))
        threading.Thread(target=_map_worker, daemon=True, name="SymbolMapping").start()
        threading.Thread(target=self._ml_retrain_worker, daemon=True).start()

        self._overview_bootstrapped = False
        self.initial_fill_symbols()
        self.root.after(100, self.refresh_overview_prices)
        self.root.after(500, self.start)
        self.root.after(2500, self.scan_ai_trader)
        self.root.after(65000, self._schedule_ai_scan)
        # FIX 6: Disabled _populate_worker to avoid double requests with main_loop
        # threading.Thread(target=self._populate_worker, daemon=True).start()

    def set_tab_activity(self, tab_name, message="آماده", state="idle"):
        """Thread-safe activity state. UI repaint is coalesced to avoid flooding Tk."""
        try:
            with TAB_ACTIVITY_LOCK:
                TAB_ACTIVITY[tab_name] = {"message": str(message), "state": str(state), "ts": time.time()}
                scheduled = bool(getattr(self, "_activity_repaint_scheduled", False))
                if scheduled:
                    return
                self._activity_repaint_scheduled = True
            self.root.after(150, self._repaint_activity_bar)
        except Exception:
            pass

    def _repaint_activity_bar(self):
        try:
            with TAB_ACTIVITY_LOCK:
                snapshot = dict(TAB_ACTIVITY)
                self._activity_repaint_scheduled = False
            widgets = getattr(self, "_tab_activity_widgets", {})
            for tab_name, item in widgets.items():
                st = snapshot.get(tab_name, {"message":"آماده", "state":"idle"})
                state = st.get("state", "idle")
                message = st.get("message", "آماده")
                color = TAB_ACTIVITY_COLORS.get(tab_name, "#94a3b8")
                state_txt = "● فعال" if state == "active" else "✓ انجام شد" if state == "done" else "○ آماده"
                fg = color if state in ("active", "done") else "#94a3b8"
                item["dot"].config(text=state_txt, fg=fg)
                item["msg"].config(text=message, fg=color if state == "active" else "#cbd5e1")
        except Exception:
            pass

    def _build_activity_bar(self):
        """Compact two-row activity monitor; avoids crowding the main toolbar."""
        parent = getattr(self, "_activity_host", self.root)
        bar = tk.Frame(parent, bg="#020617", bd=0, relief="flat", height=50)
        bar.pack(fill="both", expand=True, padx=2, pady=1)
        bar.pack_propagate(False)
        title = tk.Label(bar, text="فعالیت", bg="#020617", fg="#e2e8f0",
                         font=("Tahoma", 8, "bold"), width=6)
        title.grid(row=0, column=0, rowspan=2, padx=(5, 6), sticky="ns")
        self._tab_activity_widgets = {}
        items = [
            ("نمای کلی", "نمای کلی"), ("مرکز معاملات هوشمند", "معاملات"),
            ("آزمایشگاه تحلیل عمیق", "تحلیل عمیق"), ("🧠 اجرای هوشمند", "اجرای هوشمند"),
            ("ساختار بازار", "ساختار"), ("تحقیق هوشمند / بهینه‌سازی", "تحقیق"),
            ("Evidence / ارزیابی سیگنال", "ارزیابی")
        ]
        for idx, (key, short_name) in enumerate(items):
            row, col = divmod(idx, 4)
            color = TAB_ACTIVITY_COLORS.get(key, "#94a3b8")
            box = tk.Frame(bar, bg="#0b1220", bd=0, highlightthickness=1, highlightbackground="#1e293b")
            box.grid(row=row, column=col + 1, padx=2, pady=2, sticky="ew")
            dot = tk.Label(box, text="●", bg="#0b1220", fg="#64748b", font=("Tahoma", 7, "bold"))
            dot.pack(side="left", padx=(4, 2))
            msg = tk.Label(box, text=short_name, bg="#0b1220", fg=color, font=("Tahoma", 7, "bold"))
            msg.pack(side="left", padx=2)
            sub = tk.Label(box, text="آماده", bg="#0b1220", fg="#94a3b8", font=("Tahoma", 7))
            sub.pack(side="right", padx=(2, 4))
            self._tab_activity_widgets[key] = {"dot": dot, "msg": sub, "color": color}
        for c in range(1, 5):
            bar.grid_columnconfigure(c, weight=1)
        return bar

    def _on_tab_changed(self, event=None):
        """Tab switching must stay UI-only; never start network work here."""
        try:
            tab_id = self.notebook.select()
            label = self.notebook.tab(tab_id, "text")
            # Strip leading icons for matching our activity keys.
            self.set_tab_activity(label, "تب فعال", "active")
        except Exception:
            pass

    def _toggle_real_trading(self):
        """Explicit two-step gate for real MARGIN Auto Trade."""
        global REAL_TRADING_ENABLED, REAL_TRADING_CONFIRMATION
        try:
            if not self.real_trading_enabled.get():
                REAL_TRADING_ENABLED = False
                REAL_TRADING_CONFIRMATION = ""
                self.real_trading_confirm.set("")
                return
            if not (globals().get("_nobitex_api_key") and globals().get("_nobitex_private_key")) and not globals().get("_nobitex_token"):
                self.real_trading_enabled.set(False)
                messagebox.showwarning("REAL TRADING", "ابتدا در تب API نوبیتکس احراز هویت را ذخیره و تست کنید.")
                return
            phrase = simpledialog.askstring(
                "تأیید معاملات واقعی",
                "برای فعال‌کردن سفارش واقعی، دقیقاً عبارت REAL را وارد کنید:",
                parent=self.root,
            )
            if str(phrase or "").strip().upper() != REAL_TRADING_CONFIRM_PHRASE:
                self.real_trading_enabled.set(False)
                REAL_TRADING_ENABLED = False
                REAL_TRADING_CONFIRMATION = ""
                messagebox.showwarning("لغو شد", "عبارت تأیید صحیح نبود؛ معاملات واقعی فعال نشدند.")
                return
            REAL_TRADING_CONFIRMATION = REAL_TRADING_CONFIRM_PHRASE
            REAL_TRADING_ENABLED = True
            self.real_trading_confirm.set(REAL_TRADING_CONFIRM_PHRASE)
            # Real mode is MARGIN; leverage is sent to Nobitex.
            messagebox.showwarning(
                "⚠️ REAL TRADING فعال شد",
                "از این لحظه Auto Trade سفارش واقعی تعهدی (MARGIN) به نوبیتکس ارسال می‌کند.\n\n"
                "SL / TP / Trailing در این نسخه توسط همین برنامه پایش می‌شوند؛\n"
                "اگر برنامه یا اینترنت قطع شود، مدیریت محلی آنها متوقف می‌شود."
            )
        except Exception as exc:
            REAL_TRADING_ENABLED = False
            REAL_TRADING_CONFIRMATION = ""
            try: self.real_trading_enabled.set(False)
            except Exception: pass
            logging.exception("real trading toggle failed: %s", exc)

    def open_capital_settings(self):
        """Open editable Spot/Margin wallet-use and per-trade settings."""
        global MARGIN_WALLET_USE_PCT, MARGIN_TRADE_USDT, SPOT_WALLET_USE_PCT, SPOT_TRADE_USDT
        global MARGIN_TRADE_RLS, SPOT_TRADE_RLS, AUTO_TRADE_USDT_VALUE
        win = tk.Toplevel(self.root)
        win.title("💰 مدیریت سرمایه معاملات")
        win.geometry("700x650")
        win.minsize(650, 610)
        win.configure(bg="#0f172a")
        win.transient(self.root); win.grab_set()
        tk.Label(win, text="💰 مدیریت سرمایه و حجم معاملات", bg="#0f172a", fg="#f8fafc",
                 font=("Tahoma", 16, "bold")).pack(pady=(14, 3))
        tk.Label(win, text="برای بازارهای IRT مبلغ ریالی مستقل تنظیم می‌شود؛ ۳ میلیون تومان = ۳۰,۰۰۰,۰۰۰ ریال.",
                 bg="#0f172a", fg="#94a3b8", font=("Tahoma", 9)).pack(pady=(0, 10))
        body=tk.Frame(win,bg="#0f172a"); body.pack(fill="both",expand=True,padx=18,pady=4)
        def section(title,pct_var,usdt_var,rls_var,accent):
            box=tk.LabelFrame(body,text=title,bg="#111827",fg=accent,font=("Tahoma",11,"bold"),padx=14,pady=10)
            box.pack(fill="x",pady=6)
            r=tk.Frame(box,bg="#111827"); r.pack(fill="x",pady=4)
            tk.Label(r,text="حداکثر درصد استفاده از موجودی:",bg="#111827",fg="white",font=("Tahoma",10)).pack(side="right")
            tk.Spinbox(r,from_=1,to=100,increment=1,textvariable=pct_var,width=8,justify="center",font=("Tahoma",10)).pack(side="right",padx=8)
            tk.Label(r,text="%",bg="#111827",fg="#38bdf8",font=("Tahoma",10,"bold")).pack(side="right")
            r=tk.Frame(box,bg="#111827"); r.pack(fill="x",pady=4)
            tk.Label(r,text="معامله USDT (بازارهای USDT):",bg="#111827",fg="white",font=("Tahoma",10)).pack(side="right")
            tk.Entry(r,textvariable=usdt_var,width=12,justify="center",font=("Tahoma",10)).pack(side="right",padx=8)
            tk.Label(r,text="USDT",bg="#111827",fg="#22c55e",font=("Tahoma",10,"bold")).pack(side="right")
            r=tk.Frame(box,bg="#111827"); r.pack(fill="x",pady=4)
            tk.Label(r,text="معامله ریالی (بازارهای IRT):",bg="#111827",fg="white",font=("Tahoma",10)).pack(side="right")
            tk.Entry(r,textvariable=rls_var,width=15,justify="center",font=("Tahoma",10)).pack(side="right",padx=8)
            tk.Label(r,text="RLS",bg="#111827",fg="#fbbf24",font=("Tahoma",10,"bold")).pack(side="right")
            tk.Label(box,text="مبلغ ریالی برای هر معامله مستقل از مبلغ USDT است.",bg="#111827",fg="#94a3b8",font=("Tahoma",8)).pack(anchor="e",pady=(4,0))
        section("🟣 معاملات تعهدی (MARGIN)",self.margin_wallet_use_pct_var,self.margin_trade_usdt_var,self.margin_trade_rls_var,"#c084fc")
        section("🟢 معاملات اسپات (SPOT)",self.spot_wallet_use_pct_var,self.spot_trade_usdt_var,self.spot_trade_rls_var,"#4ade80")
        tk.Label(win,text="پیش‌فرض بازارهای ریالی: ۳۰,۰۰۰,۰۰۰ ریال (۳ میلیون تومان) برای هر معامله.",bg="#172033",fg="#fde68a",font=("Tahoma",9,"bold"),padx=12,pady=9).pack(fill="x",padx=18,pady=8)
        buttons=tk.Frame(win,bg="#0f172a"); buttons.pack(fill="x",padx=18,pady=(0,14))
        def save():
            global MARGIN_WALLET_USE_PCT,MARGIN_TRADE_USDT,SPOT_WALLET_USE_PCT,SPOT_TRADE_USDT,MARGIN_TRADE_RLS,SPOT_TRADE_RLS,AUTO_TRADE_USDT_VALUE
            try:
                mp=max(1.0,min(100.0,float(self.margin_wallet_use_pct_var.get()))); mt=max(.01,float(self.margin_trade_usdt_var.get())); mr=max(1000.0,float(self.margin_trade_rls_var.get()))
                sp=max(1.0,min(100.0,float(self.spot_wallet_use_pct_var.get()))); st=max(.01,float(self.spot_trade_usdt_var.get())); sr=max(1000.0,float(self.spot_trade_rls_var.get()))
                MARGIN_WALLET_USE_PCT,MARGIN_TRADE_USDT,MARGIN_TRADE_RLS=mp,mt,mr; SPOT_WALLET_USE_PCT,SPOT_TRADE_USDT,SPOT_TRADE_RLS=sp,st,sr
                AUTO_TRADE_USDT_VALUE=mt if not REAL_TRADING_SPOT_ONLY else st
                _save_settings(); logging.info("[CAPITAL SETTINGS] Margin %.1f%% / %.4f USDT / %.0f RLS | Spot %.1f%% / %.4f USDT / %.0f RLS",mp,mt,mr,sp,st,sr)
                messagebox.showinfo("ذخیره شد",f"تعهدی: {mp:.1f}% | {mt:.4f} USDT | {mr:,.0f} RLS\nاسپات: {sp:.1f}% | {st:.4f} USDT | {sr:,.0f} RLS",parent=win); win.destroy()
            except Exception as exc: messagebox.showerror("خطا",f"مقدار واردشده معتبر نیست.\n{exc}",parent=win)
        tk.Button(buttons,text="💾 ذخیره تنظیمات",command=save,bg="#10b981",fg="white",font=("Tahoma",10,"bold"),width=18,pady=7).pack(side="right",padx=5)
        tk.Button(buttons,text="انصراف",command=win.destroy,bg="#334155",fg="white",font=("Tahoma",10),width=12,pady=7).pack(side="left",padx=5)
        win.bind("<Escape>",lambda e:win.destroy())

    def _build_ui(self):
        ctrl = tk.Frame(self.root, bg="#222", height=80)
        ctrl.pack(fill="x", padx=8, pady=6)

        def ctrl_button(parent, text, cmd, **kw):
            width = kw.pop("width", CTRL_BTN_WIDTH)
            return tk.Button(parent, text=text, command=cmd, bg=CTRL_BTN_BG, fg=CTRL_BTN_FG,
                             width=width, font=CTRL_BTN_FONT, relief="raised", **kw)

        ctrl_button(ctrl, "شروع", self.start).pack(side="left", padx=4)
        ctrl_button(ctrl, "توقف", self.stop).pack(side="left", padx=4)
        ctrl_button(ctrl, "کپی جدول", self.copy_table).pack(side="left", padx=4)
        ctrl_button(ctrl, "Build EXE", lambda: threading.Thread(target=self.build_exe, daemon=True).start()).pack(side="left", padx=4)
        ctrl_button(ctrl, "آموزش ML", lambda: threading.Thread(target=self.train_ml_models, daemon=True).start()).pack(side="left", padx=4)
        ctrl_button(ctrl, "بک‌تست", self.run_backtest_dialog).pack(side="left", padx=4)
        ctrl_button(ctrl, "💰 مدیریت سرمایه", self.open_capital_settings, width=16).pack(side="left", padx=4)

        # ===== کنترل سریع و همیشه‌قابل‌مشاهده Trailing / Defensive Shield =====
        # این کنترل‌ها عمداً در نوار اصلی قرار گرفته‌اند تا پشت تنظیمات یا پایین صفحه پنهان نشوند.
        quick_risk = tk.LabelFrame(ctrl, text="کنترل سریع ریسک", bg="#1e293b", fg="#93c5fd",
                                   font=("Tahoma", 9, "bold"), padx=4, pady=2)
        quick_risk.pack(side="left", padx=8, pady=2)

        def _quick_adjust(var, delta, minimum, maximum, value_label):
            try:
                value = float(var.get()) + float(delta)
                value = max(float(minimum), min(float(maximum), value))
                var.set(round(value, 2))
                value_label.config(text=f"{value:.2f}%")
                _save_settings()
            except Exception as exc:
                logging.debug("quick risk adjustment failed: %s", exc)

        # Trailing activation: آستانه شروع تریلینگ
        trail_quick = tk.Frame(quick_risk, bg="#1e293b")
        trail_quick.pack(side="left", padx=4)
        tk.Label(trail_quick, text="تریلینگ شروع", bg="#1e293b", fg="white",
                 font=("Tahoma", 7, "bold")).pack(side="left", padx=2)
        trail_value_lbl = tk.Label(trail_quick, text=f"{self.trailing_activate_var.get():.2f}%",
                                   bg="#1e293b", fg="#38bdf8", width=7,
                                   font=("Tahoma", 9, "bold"))
        trail_value_lbl.pack(side="left", padx=2)
        tk.Button(trail_quick, text="−", width=2, command=lambda: _quick_adjust(
            self.trailing_activate_var, -0.1, 0.1, 10.0, trail_value_lbl),
            bg="#334155", fg="white", font=("Tahoma", 8, "bold"), padx=0, pady=0).pack(side="left", padx=1)
        tk.Button(trail_quick, text="+", width=2, command=lambda: _quick_adjust(
            self.trailing_activate_var, 0.1, 0.1, 10.0, trail_value_lbl),
            bg="#334155", fg="white", font=("Tahoma", 8, "bold"), padx=0, pady=0).pack(side="left", padx=1)

        # Defensive Shield: درصد مجاز ضرر
        shield_quick = tk.Frame(quick_risk, bg="#1e293b")
        shield_quick.pack(side="left", padx=6)
        tk.Label(shield_quick, text="سپر دفاعی", bg="#1e293b", fg="white",
                 font=("Tahoma", 7, "bold")).pack(side="left", padx=2)
        shield_value_lbl = tk.Label(shield_quick, text=f"{self.stop_loss_pct_var.get():.2f}%",
                                    bg="#1e293b", fg="#f87171", width=7,
                                    font=("Tahoma", 9, "bold"))
        shield_value_lbl.pack(side="left", padx=2)
        tk.Button(shield_quick, text="−", width=2, command=lambda: _quick_adjust(
            self.stop_loss_pct_var, -0.1, 0.1, 20.0, shield_value_lbl),
            bg="#334155", fg="white", font=("Tahoma", 8, "bold"), padx=0, pady=0).pack(side="left", padx=1)
        tk.Button(shield_quick, text="+", width=2, command=lambda: _quick_adjust(
            self.stop_loss_pct_var, 0.1, 0.1, 20.0, shield_value_lbl),
            bg="#334155", fg="white", font=("Tahoma", 8, "bold"), padx=0, pady=0).pack(side="left", padx=1)

        # هر تغییر از تنظیمات اصلی هم مقدار نمایش سریع را به‌روز کند.
        self.trailing_activate_var.trace_add("write", lambda *_: trail_value_lbl.config(
            text=f"{float(self.trailing_activate_var.get()):.2f}%"))
        self.stop_loss_pct_var.trace_add("write", lambda *_: shield_value_lbl.config(
            text=f"{float(self.stop_loss_pct_var.get()):.2f}%"))

        # نوار مستقل و برجسته کنترل‌های اصلی؛ خارج از toolbar تا هیچ کنترلی فشرده یا پنهان نشود.
        top_switch_bar = tk.Frame(self.root, bg="#0b1220", bd=1, relief="solid",
                                  highlightthickness=1, highlightbackground="#334155")
        top_switch_bar.pack(fill="x", padx=8, pady=(0, 6), ipady=5)
        tk.Label(top_switch_bar, text="کنترل‌های سریع", bg="#0b1220", fg="#e2e8f0",
                 font=("Tahoma", 10, "bold")).pack(side="right", padx=(12, 8))

        primary_switches = tk.Frame(top_switch_bar, bg="#0b1220")
        primary_switches.pack(side="right", padx=4)

        def _make_top_switch(parent, text, variable, command, bg, fg, width=18):
            card = tk.Frame(parent, bg=bg, bd=1, relief="solid",
                            highlightthickness=1, highlightbackground=fg)
            card.pack(side="right", padx=3, pady=1, ipadx=3, ipady=2)
            cb = tk.Checkbutton(
                card, text=text, variable=variable, command=command,
                bg=bg, fg=fg, selectcolor="#0f172a",
                activebackground=bg, activeforeground="#ffffff",
                font=("Tahoma", 9, "bold"), padx=5, pady=2,
                indicatoron=True, bd=0, relief="flat", cursor="hand2"
            )
            cb.pack()
            return cb

        _make_top_switch(primary_switches, "فعال‌سازی تریلینگ", self.trailing_enabled,
                         self.on_trailing_toggle, "#2e1065", "#ddd6fe")
        _make_top_switch(primary_switches, "اتو ترید", self.auto_trade,
                         self._sync_threadsafe_flags, "#052e16", "#86efac")
        _make_top_switch(primary_switches, "⚠️ REAL MARGIN", self.real_trading_enabled,
                         self._toggle_real_trading, "#7f1d1d", "#fecaca", width=20)

        secondary_switches = tk.Frame(top_switch_bar, bg="#0b1220")
        secondary_switches.pack(side="left", padx=8)
        for _text, _var, _cmd, _fg in (
            ("فعال‌سازی طلایه", self.talaye_enabled, self.toggle_talaye_enable, "#fbbf24"),
            ("ML پیشرفته", self.ml_enabled, self._sync_threadsafe_flags, "#7dd3fc"),
            ("نمایش ساختار", self.show_structure, None, "#cbd5e1"),
            ("TTS", self.tts_enabled, None, "#f9a8d4"),
        ):
            tk.Checkbutton(
                secondary_switches, text=_text, variable=_var, command=_cmd,
                bg="#0b1220", fg=_fg, selectcolor="#1e293b",
                activebackground="#0b1220", activeforeground="#ffffff",
                font=("Tahoma", 8, "bold"), padx=4, pady=2,
                bd=0, relief="flat", cursor="hand2"
            ).pack(side="left", padx=4)

        settings_frame = tk.Frame(ctrl, bg="#222")
        settings_frame.pack(side="left", padx=20)

        def create_scale_block_with_buttons(parent, title, var, frm, to, step, fmt="{:.2f}", key=None):
            lf = tk.LabelFrame(parent, text=title, bg="#1e293b", fg="#94a3b8", font=("Tahoma", 9, "bold"), padx=8, pady=6)
            lbl = tk.Label(lf, text=title, bg="#1e293b", fg="white", font=("Tahoma", 9))
            lbl.grid(row=0, column=0, padx=(0,6))
            scl = tk.Scale(lf, from_=frm, to=to, resolution=step, orient="horizontal", variable=var,
                           bg="#1e293b", fg="white", troughcolor="#334155", highlightbackground="#1e293b", activebackground=CTRL_BTN_BG, length=140)
            scl.grid(row=0, column=1, padx=6)
            val_lbl = tk.Label(lf, text=fmt.format(var.get()), bg="#1e293b", fg=CTRL_BTN_BG, font=("Tahoma", 10, "bold"), width=8)
            val_lbl.grid(row=0, column=2, padx=6)
            def on_var_change(*_):
                try:
                    val_lbl.config(text=fmt.format(var.get()))
                except:
                    val_lbl.config(text=str(var.get()))
            var.trace_add("write", on_var_change)
            def make_adj_button(symbol, delta):
                def _():
                    try:
                        val = float(var.get())
                        new = round(val + delta, 6) if isinstance(step, float) else int(val + delta)
                        if new < frm: new = frm
                        if new > to: new = to
                        var.set(new)
                    except:
                        pass
                return tk.Button(lf, text=symbol, command=_, bg=CTRL_BTN_BG, fg=CTRL_BTN_FG, width=3)
            btn_minus = make_adj_button("-", -step)
            btn_minus.grid(row=0, column=3, padx=(6,2))
            btn_plus = make_adj_button("+", step)
            btn_plus.grid(row=0, column=4)
            lf.scale = scl
            lf.val_lbl = val_lbl
            lf.btn_plus = btn_plus
            lf.btn_minus = btn_minus
            if key:
                self._scale_blocks[key] = lf
            return lf

        trail_act_block = create_scale_block_with_buttons(settings_frame, "شروع تریلینگ (%)", self.trailing_activate_var, 0.1, 10.0, 0.1, fmt="{:.2f}", key="trailing_activate")
        trail_act_block.grid(row=0, column=0, padx=5, pady=4)
        trail_stage2_block = create_scale_block_with_buttons(settings_frame, "مرحله دوم (%)", self.trailing_stage2_var, 1.1, 20.0, 0.1, fmt="{:.2f}", key="trailing_stage2")
        trail_stage2_block.grid(row=0, column=1, padx=5, pady=4)
        trail_s1_block = create_scale_block_with_buttons(settings_frame, "افت مجاز مرحله اول (%)", self.trailing_stage1_retrace_var, 0.0, 95.0, 1.0, fmt="{:.0f}", key="trailing_stage1_retrace")
        trail_s1_block.grid(row=0, column=2, padx=5, pady=4)
        trail_s2_block = create_scale_block_with_buttons(settings_frame, "افت مجاز مرحله دوم (%)", self.trailing_stage2_retrace_var, 0.0, 95.0, 1.0, fmt="{:.0f}", key="trailing_stage2_retrace")
        trail_s2_block.grid(row=0, column=3, padx=5, pady=4)
        stop_block = create_scale_block_with_buttons(settings_frame, "سپر دفاعی (%)", self.stop_loss_pct_var, 0.1, 20.0, 0.1, fmt="{:.2f}", key="stop_loss")
        stop_block.grid(row=0, column=4, padx=5, pady=4)

        # ردیف توضیحات و کنترل‌های مستقل؛ چک‌باکس قبلاً با کنترل طلایه
        # در ستون ۴ هم‌پوشانی داشت و در بعضی اندازه‌های پنجره قابل کلیک نبود.
        tk.Label(settings_frame, text="قانون: +1%→70% برگشت مجاز / +2%→50% برگشت مجاز | Trailing هرگز منفی نمی‌بندد", bg="#222", fg="#93c5fd", font=("Tahoma", 9, "bold")).grid(row=1, column=0, columnspan=6, sticky="ew", padx=8, pady=(0,5))

        trailing_chk = tk.Checkbutton(
            settings_frame, text="✓ فعال‌سازی تریلینگ", variable=self.trailing_enabled,
            bg="#123047", fg="#7dd3fc", selectcolor="#0f172a",
            activebackground="#164e63", activeforeground="white",
            font=("Tahoma", 10, "bold"), relief="ridge", bd=2,
            padx=10, pady=7, cursor="hand2", command=self.on_trailing_toggle
        )
        trailing_chk.grid(row=2, column=0, padx=8, pady=(3,7), sticky="w")

        talaye_btn_frame = tk.Frame(settings_frame, bg="#222")
        talaye_btn_frame.grid(row=2, column=1, columnspan=5, padx=6, pady=4, sticky="w")
        def small_btn(p, text, cmd):
            return tk.Button(p, text=text, command=cmd, bg="#fb8500", fg="white", width=6, font=("Tahoma", 9, "bold"))
        small_btn(talaye_btn_frame, "طلایه -", lambda: adjust_talaye(-0.05)).pack(side="left", padx=2)
        self.talaye_value_label = tk.Label(talaye_btn_frame, text=f"{TALAYE_MIN_SCORE:.3f}", bg="#222", fg="#fde68a", font=("Tahoma", 10, "bold"), width=8)
        self.talaye_value_label.pack(side="left", padx=4)
        small_btn(talaye_btn_frame, "طلایه +", lambda: adjust_talaye(+0.05)).pack(side="left", padx=2)

        # کنترل مستقیم آستانه طلایه توسط کاربر
        self.talaye_user_value = tk.DoubleVar(value=float(TALAYE_MIN_SCORE))

        def apply_talaye_user_value(*_):
            global TALAYE_MIN_SCORE, TALAYE_MIN_SCORE_V2
            try:
                value = float(self.talaye_user_value.get())
                value = max(0.05, min(1.0, value))
                with state_lock:
                    TALAYE_MIN_SCORE = round(value, 3)
                    TALAYE_MIN_SCORE_V2 = round(value, 3)
                self.talaye_user_value.set(TALAYE_MIN_SCORE)
                self.talaye_value_label.config(text=f"{TALAYE_MIN_SCORE:.3f}")
                self.lbl_talaye_threshold.config(text=f"طلایه آستانه: {TALAYE_MIN_SCORE:.3f}")
                try:
                    _save_settings()
                except Exception:
                    pass
            except Exception:
                pass

        tk.Label(talaye_btn_frame, text="مقدار:", bg="#222", fg="white").pack(side="left", padx=(8,2))
        self.talaye_spinbox = tk.Spinbox(
            talaye_btn_frame, from_=0.05, to=1.0, increment=0.01,
            textvariable=self.talaye_user_value, width=7, format="%.3f",
            command=apply_talaye_user_value, bg="#111827", fg="white",
            insertbackground="white"
        )
        self.talaye_spinbox.pack(side="left", padx=2)
        tk.Button(talaye_btn_frame, text="اعمال", command=apply_talaye_user_value,
                  bg="#10b981", fg="white", width=6).pack(side="left", padx=2)
        self.talaye_user_value.trace_add("write", lambda *_: self.root.after_idle(apply_talaye_user_value))

        settings_frame.columnconfigure(0, weight=1)
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(2, weight=1)
        settings_frame.columnconfigure(3, weight=0)
        settings_frame.columnconfigure(4, weight=0)

        self.lbl_talaye_threshold = tk.Label(ctrl, text=f"طلایه آستانه: {TALAYE_MIN_SCORE:.3f}", bg="#222", fg="white")
        self.lbl_talaye_threshold.pack(side="left", padx=8)
        self.talaye_value_entry = tk.Entry(ctrl, width=7)
        self.talaye_value_entry.insert(0, f"{TALAYE_MIN_SCORE:.3f}")
        self.talaye_value_entry.pack(side="left", padx=4)
        def _apply_talaye_manual():
            global TALAYE_MIN_SCORE, TALAYE_MIN_SCORE_V2
            try:
                v=max(0.0,min(2.0,float(self.talaye_value_entry.get().strip())))
                TALAYE_MIN_SCORE=v
                try: TALAYE_MIN_SCORE_V2=v
                except Exception: pass
                self.lbl_talaye_threshold.config(text=f"طلایه آستانه: {v:.3f}")
            except Exception:
                pass
        tk.Button(ctrl,text="اعمال",command=_apply_talaye_manual,bg="#10b981",fg="white").pack(side="left",padx=2)


        self.lbl_status = tk.Label(self.root, text="آماده", bg="#1f2937", fg="#f59e0b")
        self.lbl_status.pack(pady=2)

        # Activity monitor MUST be a sibling packed before the expanding Notebook.
        # This prevents Tk pack geometry from hiding/compressing the monitor.
        self._activity_host = tk.Frame(self.root, bg="#020617", height=56, bd=0)
        self._activity_host.pack(fill="x", padx=8, pady=(2, 4))
        self._activity_host.pack_propagate(False)

        # خوانایی و رنگ‌بندی تب‌ها؛ فونت با دکمه‌های اصلی برنامه یکسان است.
        try:
            _tab_style = ttk.Style(self.root)
            _tab_style.theme_use("clam")
            _tab_style.configure(
                "TNotebook", background="#080f1d", borderwidth=0,
                tabmargins=(2, 2, 2, 0)
            )
            _tab_style.configure(
                "TNotebook.Tab", background="#1e293b", foreground="#f1f5f9",
                padding=(9, 5), font=("Tahoma", 9, "bold"), borderwidth=0,
                lightcolor="#1e293b", darkcolor="#1e293b", relief="flat"
            )
            _tab_style.map(
                "TNotebook.Tab",
                background=[("selected", "#2563eb"), ("active", "#334155"), ("!selected", "#1e293b")],
                foreground=[("selected", "#ffffff"), ("active", "#ffffff"), ("!selected", "#e2e8f0")],
                bordercolor=[("selected", "#93c5fd"), ("!selected", "#475569")],
                expand=[("selected", (1, 1, 1, 0))]
            )
        except Exception as _style_exc:
            logging.debug("Notebook tab styling skipped: %s", _style_exc)

        self.notebook = ttk.Notebook(self.root, style="TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=8, pady=6)

        # Every notebook page is a real scrollable web-style page.
        # Tables/Text widgets keep their own internal scrollbars.
        _install_page_scroll_support(self.root)

        self._build_activity_bar()
        try:
            self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed, add="+")
        except Exception:
            pass

        # --- Overview Tab ---
        tab_overview = _make_scrollable_tab(self.notebook, bg="#0f172a")
        self.notebook.add(tab_overview._scroll_outer, text="🟢 نمای کلی")
        main = tk.Frame(tab_overview)
        main.pack(fill="both", expand=True, padx=8, pady=6)
        # OVERVIEW V10.5: one canonical schema. Every insert/update path MUST
        # produce exactly these 20 values in exactly this order.
        cols = (
            "symbol","price","change","rsi","macd","dem","bbu","bbl","vwap","volk",
            "signal","advice","score","prob","pred_move","target","mtf","tag","side","rules"
        )
        col_titles = {
            "symbol":"نماد","price":"قیمت","change":"تغییر","rsi":"RSI",
            "macd":"MACD","dem":"DeM","bbu":"BB بالا","bbl":"BB پایین",
            "vwap":"VWAP","volk":"VolK","signal":"Signal","advice":"Advice",
            "score":"Score","prob":"Prob","pred_move":"PredMove","target":"Target",
            "mtf":"MTF","tag":"Tag","side":"Side","rules":"Rules"
        }
        table_frame = tk.Frame(main)
        table_frame.pack(fill="both", expand=True, side="left")
        self.main_table = ttk.Treeview(table_frame, columns=cols, show="headings")
        # Defensive invariant: every Overview row must have exactly 20 cells.
        self._overview_column_count = len(cols)
        vsb_main = ttk.Scrollbar(table_frame, orient="vertical", command=self.main_table.yview)
        hsb_main = ttk.Scrollbar(table_frame, orient="horizontal", command=self.main_table.xview)
        self.main_table.configure(yscrollcommand=vsb_main.set, xscrollcommand=hsb_main.set)
        for c in cols:
            self.main_table.heading(c, text=col_titles.get(c, c), command=lambda _c=c: self.sort_tree(self.main_table, _c))
            self.main_table.column(c, width=110, anchor="center")
        self.main_table.grid(row=0, column=0, sticky="nsew")
        vsb_main.grid(row=0, column=1, sticky="ns")
        hsb_main.grid(row=1, column=0, sticky="ew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        self.main_table.tag_configure("long_strong", background="#166534", foreground="white")
        self.main_table.tag_configure("long", background="#15803d", foreground="white")
        self.main_table.tag_configure("short_strong", background="#991b1b", foreground="white")
        self.main_table.tag_configure("short", background="#dc2626", foreground="white")
        self.main_table.tag_configure("neutral", background="#374151", foreground="#9ca3af")
        self.main_table.tag_configure("talaye_on", background="#FFD700", foreground="black")
        self.main_table.tag_configure("talaye_off", background="#FFDF80", foreground="black")
        self.lbl_selected_symbol = tk.Label(
            main, text="نماد انتخاب‌شده: —",
            bg="#111827", fg="#38bdf8", font=("Tahoma", 10, "bold")
        )
        self.lbl_selected_symbol.pack(fill="x", before=table_frame, pady=(0, 4))

        def _on_overview_select(_event=None):
            try:
                sel = self.main_table.selection()
                if sel:
                    vals = self.main_table.item(sel[0]).get("values", [])
                    if vals:
                        self.lbl_selected_symbol.config(
                            text=f"نماد انتخاب‌شده: {vals[0]}"
                        )
                        self.structure_symbol_combo.set(str(vals[0]))
                        self.refresh_structure_tab()
            except Exception:
                pass

        self.main_table.bind("<<TreeviewSelect>>", _on_overview_select)
        _install_scroll_support(self.main_table)


        # --- Unified Trading Center: Signals + Auto Trade + AI Trader ---
        # All three engines now live in ONE outer tab.  The underlying widgets
        # keep their original names so the existing worker/refresh methods
        # continue to work, while the user gets a single operational dashboard.
        tab_trading = _make_scrollable_tab(self.notebook, bg="#0b1220")
        self.notebook.add(tab_trading._scroll_outer, text="🔵 مرکز معاملات هوشمند")
        self.trading_center_tab = tab_trading

        trading_top = tk.Frame(tab_trading, bg="#111827", bd=0)
        trading_top.pack(fill="x", padx=6, pady=(5, 3))

        tk.Label(
            trading_top, text="مرکز معاملات هوشمند — Signal + AI Trader + Auto Trade",
            bg="#111827", fg="#38bdf8", font=("Tahoma", 12, "bold")
        ).pack(side="left", padx=8)

        self.ai_status = tk.Label(trading_top, text="آماده", bg="#111827", fg="#fbbf24")
        self.ai_status.pack(side="left", padx=10)

        ctrl_button(trading_top, "اسکن AI", self.scan_ai_trader).pack(side="left", padx=3)
        ctrl_button(trading_top, "به‌روزرسانی سیگنال", self.refresh_tables).pack(side="left", padx=3)
        ctrl_button(trading_top, "کپی سیگنال", self.copy_signals).pack(side="left", padx=3)
        ctrl_button(trading_top, "Export CSV", self.export_signals).pack(side="left", padx=3)
        ctrl_button(trading_top, "پاک‌کردن سیگنال", self.clear_signal_log).pack(side="left", padx=3)
        ctrl_button(trading_top, "پاک‌کردن معاملات", self.clear_auto_trade_log).pack(side="left", padx=3)

        # کنترل‌های Auto Trade و Trailing فقط در نوار ثابت بالای پنجره قرار دارند.
        self.center_auto_var = self.auto_trade
        self.center_trailing_var = self.trailing_enabled

        center_stats = tk.Frame(tab_trading, bg="#0b1220")
        center_stats.pack(fill="x", padx=6, pady=(0, 4))
        self.lbl_center_signals = tk.Label(center_stats, text="سیگنال: 0", bg="#172033", fg="#22c55e", padx=12, pady=4)
        self.lbl_center_open = tk.Label(center_stats, text="معامله باز: 0", bg="#172033", fg="#a78bfa", padx=12, pady=4)
        self.lbl_center_ai = tk.Label(center_stats, text="AI: آماده", bg="#172033", fg="#60a5fa", padx=12, pady=4)
        self.lbl_center_risk = tk.Label(center_stats, text="Risk: —", bg="#172033", fg="#f59e0b", padx=12, pady=4)
        for w in (self.lbl_center_signals, self.lbl_center_open, self.lbl_center_ai, self.lbl_center_risk):
            w.pack(side="left", padx=3)

        center_body = tk.Frame(tab_trading, bg="#0b1220")
        center_body.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        center_body.grid_rowconfigure(0, weight=5)
        center_body.grid_rowconfigure(1, weight=4)
        center_body.grid_columnconfigure(0, weight=1)
        center_body.grid_columnconfigure(1, weight=1)

        # ===== AI / opportunity table (top) =====
        ai_box = tk.LabelFrame(
            center_body, text="🤖 AI Trader — فرصت‌های زنده و تصمیم چندلایه",
            bg="#0f172a", fg="#60a5fa", font=("Tahoma", 10, "bold")
        )
        ai_box.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
        ai_box.grid_rowconfigure(1, weight=1); ai_box.grid_columnconfigure(0, weight=1)

        ai_head = tk.Frame(ai_box, bg="#111827")
        ai_head.grid(row=0, column=0, sticky="ew", padx=4, pady=3)
        tk.Label(ai_head, text="BUY++ / SELL++ = تأیید قوی | BUY / SELL = تأیید اولیه | WAIT = بدون تأیید کافی",
                 bg="#111827", fg="#cbd5e1").pack(side="left", padx=6)
        tk.Label(ai_head, text="آستانه AI:", bg="#111827", fg="white").pack(side="left", padx=(15, 3))
        self.ai_threshold_var = tk.DoubleVar(value=AI_MIN_SCORE)
        self.ai_threshold_entry = tk.Spinbox(ai_head, from_=50.0, to=95.0, increment=1.0,
                                             textvariable=self.ai_threshold_var, width=6)
        self.ai_threshold_entry.pack(side="left", padx=2)
        def apply_ai_threshold():
            global AI_MIN_SCORE, AI_MIN_SCORE_STRONG
            try:
                v=max(50.0,min(95.0,float(self.ai_threshold_var.get())))
                AI_MIN_SCORE=v; AI_MIN_SCORE_STRONG=min(95.0,max(v+8,78.0)); self.ai_threshold_var.set(v)
                self.ai_status.config(text=f"حد AI = {v:.0f}")
            except Exception: pass
        tk.Button(ai_head, text="اعمال", command=apply_ai_threshold, bg="#10b981", fg="white").pack(side="left", padx=3)

        ai_table_host = tk.Frame(ai_box, bg="#0f172a")
        ai_table_host.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)
        ai_table_host.grid_rowconfigure(0, weight=1); ai_table_host.grid_columnconfigure(0, weight=1)
        ai_cols = (
            "symbol","signal","score","confidence","regime","rsi","macd","adx","stoch","cci","mfi","willr",
            "bb","vwap","ema","volume","mtf","method1","method2","method3","support","resistance","entry","sl","tp1","tp2","rr"
        )
        ai_titles = {
            "symbol":"نماد","signal":"سیگنال","score":"امتیاز","confidence":"اعتماد","regime":"رژیم","rsi":"RSI","macd":"MACD",
            "adx":"ADX","stoch":"Stoch","cci":"CCI","mfi":"MFI","willr":"Williams %R","bb":"Bollinger","vwap":"VWAP",
            "ema":"EMA 9/21/50","volume":"حجم","mtf":"MTF","method1":"روش۱","method2":"روش۲","method3":"روش۳",
            "support":"حمایت","resistance":"مقاومت","entry":"ورود","sl":"حدضرر","tp1":"TP1","tp2":"TP2","rr":"R/R"
        }
        self.ai_table = ttk.Treeview(ai_table_host, columns=ai_cols, show="headings", height=8)
        ai_v=ttk.Scrollbar(ai_table_host,orient="vertical",command=self.ai_table.yview)
        ai_h=ttk.Scrollbar(ai_table_host,orient="horizontal",command=self.ai_table.xview)
        self.ai_table.configure(yscrollcommand=ai_v.set,xscrollcommand=ai_h.set)
        self.ai_table.grid(row=0,column=0,sticky="nsew"); ai_v.grid(row=0,column=1,sticky="ns"); ai_h.grid(row=1,column=0,sticky="ew")
        for c in ai_cols:
            self.ai_table.heading(c,text=ai_titles[c]); self.ai_table.column(c,width=92,anchor="center")
        self.ai_table.tag_configure("buypp",background="#166534",foreground="white")
        self.ai_table.tag_configure("buy",background="#15803d",foreground="white")
        self.ai_table.tag_configure("sellpp",background="#991b1b",foreground="white")
        self.ai_table.tag_configure("sell",background="#dc2626",foreground="white")
        self.ai_table.tag_configure("wait",background="#374151",foreground="#cbd5e1")
        _install_scroll_support(self.ai_table)
        ai_report_host=tk.Frame(ai_box,bg="#0b1220",height=54); ai_report_host.grid(row=2,column=0,sticky="ew",padx=4,pady=(2,3)); ai_report_host.grid_propagate(False)
        self.ai_report=tk.Text(ai_report_host,height=3,bg="#0b1220",fg="#cbd5e1",wrap="none",font=("Consolas",9))
        self.ai_report.pack(fill="both",expand=True)
        _install_scroll_support(self.ai_report)

        # ===== Signals table (bottom-left) =====
        sig_box=tk.LabelFrame(center_body,text="📡 سیگنال‌ها — تاریخچه، اعتبار، هدف و وضعیت",bg="#0f172a",fg="#22c55e",font=("Tahoma",10,"bold"))
        sig_box.grid(row=1,column=0,sticky="nsew",padx=(2,1),pady=2)
        sig_box.grid_rowconfigure(1,weight=1); sig_box.grid_columnconfigure(0,weight=1)
        sig_tools=tk.Frame(sig_box,bg="#111827"); sig_tools.grid(row=0,column=0,sticky="ew",padx=3,pady=2)
        tk.Label(sig_tools,text="Double-click = ویرایش/جزئیات",bg="#111827",fg="#9ca3af").pack(side="left",padx=5)
        self.signals_frame=tk.Frame(sig_box,bg="#0f172a"); self.signals_frame.grid(row=1,column=0,sticky="nsew",padx=3,pady=2)
        self.signals_frame.grid_rowconfigure(0,weight=1); self.signals_frame.grid_columnconfigure(0,weight=1)
        sig_cols=("id","symbol","time","entry","exit","exit_time","current_price","live_pct","result","side","score","confidence","talaye","talaye_score","matched_rules","manual_entry","manual_target_pct","manual_target_price","predicted_move_pct","remaining_pct","target_price","expire_time","expired","action")
        sig_headings={"id":"شناسه","symbol":"نماد","time":"زمان","entry":"ورود","exit":"خروج","exit_time":"زمان خروج","current_price":"قیمت فعلی","live_pct":"P/L%","result":"نتیجه","side":"پوزیشن","score":"امتیاز","confidence":"اعتماد","talaye":"طلایه","talaye_score":"طلایه امتیاز","matched_rules":"قوانین","manual_entry":"ورود دستی","manual_target_pct":"هدف%","manual_target_price":"قیمت هدف","predicted_move_pct":"پیش‌بینی%","remaining_pct":"باقیمانده%","target_price":"هدف","expire_time":"انقضا","expired":"منقضی","action":"فروش فوری / بستن"}
        self.signals_table=ttk.Treeview(self.signals_frame,columns=sig_cols,show="headings",height=6)
        vs=ttk.Scrollbar(self.signals_frame,orient="vertical",command=self.signals_table.yview); hs=ttk.Scrollbar(self.signals_frame,orient="horizontal",command=self.signals_table.xview)
        self.signals_table.configure(yscrollcommand=vs.set,xscrollcommand=hs.set)
        self.signals_table.grid(row=0,column=0,sticky="nsew"); vs.grid(row=0,column=1,sticky="ns"); hs.grid(row=1,column=0,sticky="ew")
        for c in sig_cols:
            self.signals_table.heading(c,text=sig_headings.get(c,c),command=lambda _c=c:self.sort_tree(self.signals_table,_c)); self.signals_table.column(c,width=95,anchor="center")
        self.signals_table.bind("<Double-1>",self.on_signal_double_click)
        self.signals_table.bind("<Button-1>", self._signal_action_click)
        self.signals_table.tag_configure("action", background="#7f1d1d", foreground="white")
        self.lbl_long_wins=tk.Label(sig_box,text="لانگ: 0/0",fg="white",bg="#111827"); self.lbl_short_wins=tk.Label(sig_box,text="شورت: 0/0",fg="white",bg="#111827")
        self.lbl_long_wins.grid(row=2,column=0,sticky="w",padx=5); self.lbl_short_wins.grid(row=2,column=0,sticky="e",padx=5)
        _install_scroll_support(self.signals_table)

        # ===== Auto-trade table (bottom-right) =====
        at_box=tk.LabelFrame(center_body,text="⚙️ Auto Trade — Simulation / REAL MARGIN + Risk + Trailing",bg="#0f172a",fg="#a78bfa",font=("Tahoma",10,"bold"))
        at_box.grid(row=1,column=1,sticky="nsew",padx=(1,2),pady=2)
        at_box.grid_rowconfigure(1,weight=1); at_box.grid_columnconfigure(0,weight=1)
        at_tools=tk.Frame(at_box,bg="#111827"); at_tools.grid(row=0,column=0,sticky="ew",padx=3,pady=2)
        tk.Label(at_tools,text="Trailing بر اساس تنظیمات ریسک بالای برنامه مدیریت می‌شود",bg="#111827",fg="#c4b5fd").pack(side="left",padx=5)
        self.autotrade_table=ttk.Treeview(at_box,columns=("time","symbol","direction","amount","usdt_value","leverage","entry_price","current_price","peak","trailing_stop","stop_loss","tp1","tp2","exit_price","exit_time","live_pct","final_pct","status","result","risk_pct","action"),show="headings",height=6)
        at_v=ttk.Scrollbar(at_box,orient="vertical",command=self.autotrade_table.yview); at_h=ttk.Scrollbar(at_box,orient="horizontal",command=self.autotrade_table.xview)
        self.autotrade_table.configure(yscrollcommand=at_v.set,xscrollcommand=at_h.set)
        self.autotrade_table.grid(row=1,column=0,sticky="nsew",padx=3); at_v.grid(row=1,column=1,sticky="ns"); at_h.grid(row=2,column=0,sticky="ew",padx=3)
        at_titles={"time":"زمان ورود","symbol":"نماد","direction":"جهت","amount":"مقدار","usdt_value":"ارزش","leverage":"اهرم","entry_price":"ورود","current_price":"قیمت فعلی","peak":"اوج","trailing_stop":"Trailing","stop_loss":"حدضرر","tp1":"TP1","tp2":"TP2","exit_price":"خروج","exit_time":"زمان خروج","live_pct":"P/L%","final_pct":"نهایی%","status":"وضعیت","result":"نتیجه","risk_pct":"ریسک%","action":"حذف"}
        for c in self.autotrade_table["columns"]:
            self.autotrade_table.heading(c,text=at_titles.get(c,c)); self.autotrade_table.column(c,width=92,anchor="center")
        self.autotrade_table.tag_configure("trail_profit",background="#166534",foreground="white"); self.autotrade_table.tag_configure("loss",background="#991b1b",foreground="white"); self.autotrade_table.tag_configure("stop_loss",background="#b45309",foreground="white"); self.autotrade_table.tag_configure("stop_loss_slip",background="#7c2d12",foreground="white"); self.autotrade_table.tag_configure("win",background="#15803d",foreground="white"); self.autotrade_table.tag_configure("open",background="#374151",foreground="white")
        self.autotrade_table.bind("<Button-1>", self._autotrade_action_click)
        _install_scroll_support(self.autotrade_table)

        # Compatibility aliases used by older refresh/copy routines.
        self.ai_body = ai_table_host
        self.signals_frame = self.signals_frame

        # --- Dedicated live Trade Monitor ---
        self._build_trade_monitor_tab()

        # Initial unified counters; refreshed whenever one of the engines updates.
        self._refresh_trading_center_summary()


        # ===== Performance Analytics =====
        tab_stats = _make_scrollable_tab(self.notebook, bg="#0b1220")
        self.notebook.add(tab_stats._scroll_outer, text="🟣 آمار معاملات")
        stats_top = tk.Frame(tab_stats, bg="#111827"); stats_top.pack(fill="x", padx=8, pady=6)
        tk.Label(stats_top, text="بازه:", bg="#111827", fg="white").pack(side="left", padx=4)
        self.stats_period = ttk.Combobox(stats_top, values=["روزانه","هفتگی","ماهانه","سالانه","همه"], state="readonly", width=10)
        self.stats_period.set("روزانه"); self.stats_period.pack(side="left", padx=4)
        tk.Label(stats_top, text="رمزارز:", bg="#111827", fg="white").pack(side="left", padx=4)
        self.stats_symbol = ttk.Combobox(stats_top, values=["همه"] + list(SYMBOLS), state="readonly", width=18); self.stats_symbol.set("همه"); self.stats_symbol.pack(side="left", padx=4)
        tk.Label(stats_top, text="نتیجه:", bg="#111827", fg="white").pack(side="left", padx=4)
        self.stats_result = ttk.Combobox(stats_top, values=["همه","TP2","تریلینگ","سپر دفاعی","درست","غلط"], state="readonly", width=12); self.stats_result.set("همه"); self.stats_result.pack(side="left", padx=4)
        ctrl_button(stats_top, "به‌روزرسانی", self.refresh_stats_dashboard).pack(side="left", padx=6)
        ctrl_button(stats_top, "Export CSV", self.export_trade_history).pack(side="left", padx=3)

        cards=tk.Frame(tab_stats,bg="#0b1220"); cards.pack(fill="x",padx=8,pady=4)
        self.stats_cards=[]
        for title in ("تعداد معاملات","تعداد سود","تعداد ضرر","مبلغ سود","مبلغ ضرر","سود/زیان خالص","Win Rate","میانگین هر معامله"):
            f=tk.Frame(cards,bg="#172033",bd=1,relief="solid"); f.pack(side="left",fill="x",expand=True,padx=2)
            tk.Label(f,text=title,bg="#172033",fg="#94a3b8",font=("Tahoma",9)).pack(pady=(5,0))
            v=tk.Label(f,text="0",bg="#172033",fg="white",font=("Tahoma",11,"bold")); v.pack(pady=(2,6)); self.stats_cards.append(v)

        body=tk.Frame(tab_stats,bg="#0b1220"); body.pack(fill="both",expand=True,padx=8,pady=5); body.grid_rowconfigure(1,weight=1); body.grid_columnconfigure(0,weight=1)
        self.stats_text=tk.Text(body,bg="#0f172a",fg="#dbeafe",font=("Consolas",10),height=9); self.stats_text.grid(row=0,column=0,sticky="ew",pady=3)
        box=tk.LabelFrame(body,text="فیلتر و رتبه‌بندی رمزارزها",bg="#0f172a",fg="#38bdf8",font=("Tahoma",10,"bold")); box.grid(row=1,column=0,sticky="nsew",pady=3); box.grid_rowconfigure(0,weight=1); box.grid_columnconfigure(0,weight=1)
        cols=("symbol","trades","wins","losses","pnl","avg","winrate","best","worst")
        self.stats_table=ttk.Treeview(box,columns=cols,show="headings")
        titles={"symbol":"رمزارز","trades":"تعداد معامله","wins":"سود","losses":"ضرر","pnl":"سود/زیان USDT","avg":"میانگین","winrate":"Win Rate","best":"بیشترین سود","worst":"بیشترین ضرر"}
        for c in cols: self.stats_table.heading(c,text=titles[c]); self.stats_table.column(c,width=125,anchor="center")
        sv=ttk.Scrollbar(box,orient="vertical",command=self.stats_table.yview); self.stats_table.configure(yscrollcommand=sv.set); self.stats_table.grid(row=0,column=0,sticky="nsew"); sv.grid(row=0,column=1,sticky="ns")
        self.stats_table.tag_configure("profit",background="#14532d",foreground="white"); self.stats_table.tag_configure("loss",background="#7f1d1d",foreground="white")
        self.stats_period.bind("<<ComboboxSelected>>", lambda e:self.refresh_stats_dashboard()); self.stats_symbol.bind("<<ComboboxSelected>>", lambda e:self.refresh_stats_dashboard()); self.stats_result.bind("<<ComboboxSelected>>", lambda e:self.refresh_stats_dashboard())
        self.root.after(1200, self.refresh_stats_dashboard)

        # --- Deep Research Lab ---
        tab_deep = tk.Frame(self.notebook)
        # MERGED INTO UNIFIED MARKET INTELLIGENCE CENTER: no standalone tab.
        top_deep = tk.Frame(tab_deep, bg="#1e293b")
        top_deep.pack(fill="x", padx=8, pady=4)
        tk.Label(top_deep, text="تحلیل عمیق ۲ ساله و چندتایم‌فریمی",
                 fg="#f59e0b", bg="#1e293b",
                 font=("Tahoma", 12, "bold")).pack(side="left")
        self.deep_progress = tk.Label(top_deep, text="آماده",
                                      fg="#fbbf24", bg="#1e293b")
        self.deep_progress.pack(side="left", padx=12)
        self.deep_symbol_combo = ttk.Combobox(top_deep, values=list(SYMBOLS),
                                              state="readonly", width=18)
        if SYMBOLS: self.deep_symbol_combo.set(SYMBOLS[0])
        self.deep_symbol_combo.pack(side="left", padx=5)
        ctrl_button(top_deep, "تحلیل نماد", self.run_deep_selected).pack(side="left", padx=3)
        ctrl_button(top_deep, "تحلیل همه", self.run_deep_all).pack(side="left", padx=3)

        deep_body=tk.Frame(tab_deep); deep_body.pack(fill="both",expand=True,padx=8,pady=6)
        deep_left=tk.Frame(deep_body,width=190); deep_left.pack(side="left",fill="y",padx=(0,8))
        tk.Label(deep_left,text="نمادها",bg="#111827",fg="white").pack(fill="x")
        self.deep_symbol_list=tk.Listbox(deep_left,bg="#0f172a",fg="#e2e8f0")
        dsv=ttk.Scrollbar(deep_left,orient="vertical",command=self.deep_symbol_list.yview)
        self.deep_symbol_list.configure(yscrollcommand=dsv.set)
        self.deep_symbol_list.pack(side="left",fill="both",expand=True); dsv.pack(side="right",fill="y")
        for _s in SYMBOLS:self.deep_symbol_list.insert("end",_s)
        self.deep_symbol_list.bind("<<ListboxSelect>>",
                                   lambda e:self.deep_symbol_combo.set(self.deep_symbol_list.get(self.deep_symbol_list.curselection()[0]))
                                   if self.deep_symbol_list.curselection() else None)
        _install_scroll_support(self.deep_symbol_list)

        deep_right=tk.Frame(deep_body); deep_right.pack(side="left",fill="both",expand=True)
        self.deep_table=ttk.Treeview(deep_right,
            columns=("symbol","score","best","5","15","60","240","1440","result"),
            show="headings")
        dv=ttk.Scrollbar(deep_right,orient="vertical",command=self.deep_table.yview)
        dh=ttk.Scrollbar(deep_right,orient="horizontal",command=self.deep_table.xview)
        self.deep_table.configure(yscrollcommand=dv.set,xscrollcommand=dh.set)
        for c,t in {"symbol":"نماد","score":"امتیاز","best":"بهترین TF","5":"۵دقیقه",
                    "15":"۱۵دقیقه","60":"۱ساعت","240":"۴ساعت","1440":"روزانه","result":"نتیجه"}.items():
            self.deep_table.heading(c,text=t); self.deep_table.column(c,width=110,anchor="center")
        self.deep_table.grid(row=0,column=0,sticky="nsew"); dv.grid(row=0,column=1,sticky="ns"); dh.grid(row=1,column=0,sticky="ew")
        deep_right.grid_rowconfigure(0,weight=1); deep_right.grid_columnconfigure(0,weight=1)
        _install_scroll_support(self.deep_table)
        self.deep_report=tk.Text(deep_right,height=18,bg="#0b1220",fg="#e5e7eb",wrap="none")
        drv=ttk.Scrollbar(deep_right,orient="vertical",command=self.deep_report.yview)
        drh=ttk.Scrollbar(deep_right,orient="horizontal",command=self.deep_report.xview)
        self.deep_report.configure(yscrollcommand=drv.set,xscrollcommand=drh.set)
        self.deep_report.grid(row=2,column=0,sticky="nsew",pady=(7,0)); drv.grid(row=2,column=1,sticky="ns",pady=(7,0)); drh.grid(row=3,column=0,sticky="ew")
        deep_right.grid_rowconfigure(2,weight=1); _install_scroll_support(self.deep_report)


        # --- Smart Execution / SMC / 100 Trades ---
        tab_smart=tk.Frame(self.notebook)
        # MERGED INTO UNIFIED MARKET INTELLIGENCE CENTER: no standalone tab.
        top=tk.Frame(tab_smart,bg="#1e293b"); top.pack(fill="x",padx=8,pady=4)
        tk.Label(top,text="انتخاب هوشمند معامله + SMC + کندل + NR4/NR7",fg="#22d3ee",bg="#1e293b",font=("Tahoma",12,"bold")).pack(side="left")
        self.smart_status=tk.Label(top,text="آماده",fg="#fbbf24",bg="#1e293b"); self.smart_status.pack(side="left",padx=10)
        ctrl_button(top,"اسکن هوشمند بازار",self.scan_smart_market).pack(side="left",padx=4)
        ctrl_button(top,"منابع تحقیقاتی",self.show_researched_sources).pack(side="left",padx=4)
        ctrl_button(top,"اسکرین‌شات سیگنال",self.save_selected_signal_snapshot).pack(side="left",padx=4)
        tk.Label(top,text="منابع: Narrow Range + TradingView Signals + Indicator Confluence",fg="#cbd5e1",bg="#1e293b").pack(side="left",padx=8)
        tk.Label(top,text="تا 100 معامله باز مستقل | بدون تداخل نمادی",fg="#cbd5e1",bg="#1e293b").pack(side="left",padx=10)

        body=tk.Frame(tab_smart); body.pack(fill="both",expand=True,padx=8,pady=6)
        cols=("symbol","signal","score","confidence","entry","sl","tp1","tp2","rr","choch","eqh","eql","fvg","imbalance","candles","nr","ob","liquidity")
        titles={"symbol":"نماد","signal":"سیگنال","score":"امتیاز","confidence":"اعتماد","entry":"ورود","sl":"حدضرر","tp1":"TP1","tp2":"TP2","rr":"R/R","choch":"CHoCH","eqh":"EQH","eql":"EQL","fvg":"FVG","imbalance":"عدم‌تعادل","candles":"کندل","nr":"NR4/NR7","ob":"Order Block","liquidity":"نقدینگی"}
        self.smart_table=ttk.Treeview(body,columns=cols,show="headings")
        sv=ttk.Scrollbar(body,orient="vertical",command=self.smart_table.yview); sh=ttk.Scrollbar(body,orient="horizontal",command=self.smart_table.xview)
        self.smart_table.configure(yscrollcommand=sv.set,xscrollcommand=sh.set)
        for c in cols:
            self.smart_table.heading(c,text=titles[c]); self.smart_table.column(c,width=125,anchor="center")
        self.smart_table.grid(row=0,column=0,sticky="nsew"); sv.grid(row=0,column=1,sticky="ns"); sh.grid(row=1,column=0,sticky="ew")
        body.grid_rowconfigure(0,weight=1); body.grid_columnconfigure(0,weight=1)
        _install_scroll_support(self.smart_table)
        self.smart_report=tk.Text(body,height=11,bg="#0b1220",fg="#e5e7eb",wrap="none")
        rv=ttk.Scrollbar(body,orient="vertical",command=self.smart_report.yview); rh=ttk.Scrollbar(body,orient="horizontal",command=self.smart_report.xview)
        self.smart_report.configure(yscrollcommand=rv.set,xscrollcommand=rh.set)
        self.smart_report.grid(row=2,column=0,sticky="nsew",pady=(7,0)); rv.grid(row=2,column=1,sticky="ns",pady=(7,0)); rh.grid(row=3,column=0,sticky="ew")
        body.grid_rowconfigure(2,weight=1); _install_scroll_support(self.smart_report)

        # --- Market Structure Tab ---
        tab_structure = tk.Frame(self.notebook)
        # MERGED INTO UNIFIED MARKET INTELLIGENCE CENTER: no standalone tab.
        struct_top = tk.Frame(tab_structure, bg="#1e293b")
        struct_top.pack(fill="x", padx=8, pady=4)
        tk.Label(
            struct_top, text="تحلیل جامع ساختار بازار",
            fg="#f59e0b", bg="#1e293b",
            font=("Tahoma", 12, "bold")
        ).pack(side="left")

        tk.Label(
            struct_top, text="نماد:",
            fg="white", bg="#1e293b"
        ).pack(side="left", padx=(15, 4))

        self.structure_symbol_combo = ttk.Combobox(
            struct_top,
            values=list(SYMBOLS),
            state="readonly",
            width=18
        )
        if SYMBOLS:
            self.structure_symbol_combo.set(SYMBOLS[0])
        self.structure_symbol_combo.pack(side="left", padx=4)

        ctrl_button(
            struct_top, "تحلیل / بروزرسانی",
            self.refresh_structure_tab
        ).pack(side="left", padx=8)

        struct_wrap = tk.Frame(tab_structure)
        struct_wrap.pack(fill="both", expand=True, padx=8, pady=6)
        self.structure_text = tk.Text(
            struct_wrap, bg="#0f172a", fg="#e2e8f0",
            font=("Consolas", 10), wrap="none"
        )
        struct_v = ttk.Scrollbar(
            struct_wrap, orient="vertical", command=self.structure_text.yview
        )
        struct_h = ttk.Scrollbar(
            struct_wrap, orient="horizontal", command=self.structure_text.xview
        )
        self.structure_text.configure(
            yscrollcommand=struct_v.set, xscrollcommand=struct_h.set
        )
        self.structure_text.grid(row=0, column=0, sticky="nsew")
        struct_v.grid(row=0, column=1, sticky="ns")
        struct_h.grid(row=1, column=0, sticky="ew")
        struct_wrap.grid_rowconfigure(0, weight=1)
        struct_wrap.grid_columnconfigure(0, weight=1)
        _install_scroll_support(self.structure_text)
        self.structure_text.insert(
            "1.0",
            "نماد را انتخاب و روی «تحلیل / بروزرسانی» کلیک کنید."
        )

        # --- Risk Dashboard Tab ---
        tab_risk = tk.Frame(self.notebook)
        # MERGED INTO UNIFIED MARKET INTELLIGENCE CENTER: no standalone tab.
        risk_top = tk.Frame(tab_risk, bg="#1e293b")
        risk_top.pack(fill="x", padx=8, pady=4)
        tk.Label(
            risk_top, text="داشبورد مدیریت ریسک",
            fg="#ef4444", bg="#1e293b",
            font=("Tahoma", 12, "bold")
        ).pack(side="left")
        ctrl_button(
            risk_top, "شبیه‌سازی مونت‌کارلو", self.run_monte_carlo
        ).pack(side="left", padx=10)

        risk_wrap = tk.Frame(tab_risk)
        risk_wrap.pack(fill="both", expand=True, padx=8, pady=6)
        self.risk_table = ttk.Treeview(
            risk_wrap,
            columns=("metric","value","status"),
            show="headings"
        )
        risk_v = ttk.Scrollbar(
            risk_wrap, orient="vertical", command=self.risk_table.yview
        )
        risk_h = ttk.Scrollbar(
            risk_wrap, orient="horizontal", command=self.risk_table.xview
        )
        self.risk_table.configure(
            yscrollcommand=risk_v.set, xscrollcommand=risk_h.set
        )
        risk_titles = {
            "metric":"شاخص","value":"مقدار","status":"وضعیت"
        }
        for c in ("metric","value","status"):
            self.risk_table.heading(
                c, text=risk_titles.get(c, c)
            )
            self.risk_table.column(c, width=240, anchor="center")
        self.risk_table.grid(row=0, column=0, sticky="nsew")
        risk_v.grid(row=0, column=1, sticky="ns")
        risk_h.grid(row=1, column=0, sticky="ew")
        risk_wrap.grid_rowconfigure(0, weight=1)
        risk_wrap.grid_columnconfigure(0, weight=1)
        _install_scroll_support(self.risk_table)

        # --- Backtest Tab ---
        tab_backtest = tk.Frame(self.notebook)
        # MERGED INTO UNIFIED MARKET INTELLIGENCE CENTER: no standalone tab.
        bt_top = tk.Frame(tab_backtest, bg="#1e293b")
        bt_top.pack(fill="x", padx=8, pady=4)

        tk.Label(
            bt_top, text="موتور بک‌تست",
            fg="#10b981", bg="#1e293b",
            font=("Tahoma", 12, "bold")
        ).pack(side="left")

        tk.Label(
            bt_top, text="انتخاب نماد:",
            fg="white", bg="#1e293b"
        ).pack(side="left", padx=(15, 4))

        self.backtest_symbol_combo = ttk.Combobox(
            bt_top, values=list(SYMBOLS),
            state="readonly", width=18
        )
        if SYMBOLS:
            self.backtest_symbol_combo.set(SYMBOLS[0])
        self.backtest_symbol_combo.pack(side="left", padx=4)

        tk.Label(
            bt_top, text="تعداد کندل:",
            fg="white", bg="#1e293b"
        ).pack(side="left", padx=(10, 4))

        self.backtest_bars_entry = tk.Entry(bt_top, width=8)
        self.backtest_bars_entry.insert(0, "2000")
        self.backtest_bars_entry.pack(side="left", padx=4)

        ctrl_button(
            bt_top, "اجرای بک‌تست", self.run_backtest_dialog
        ).pack(side="left", padx=8)

        self.bt_progress = tk.Label(
            bt_top, text="آماده",
            fg="#fbbf24", bg="#1e293b"
        )
        self.bt_progress.pack(side="left", padx=8)

        bt_body = tk.Frame(tab_backtest)
        bt_body.pack(fill="both", expand=True, padx=8, pady=6)

        bt_left = tk.Frame(bt_body, width=190)
        bt_left.pack(side="left", fill="y", padx=(0, 8))
        tk.Label(
            bt_left, text="همه نمادهای موجود",
            bg="#111827", fg="white",
            font=("Tahoma", 10, "bold")
        ).pack(fill="x")

        self.backtest_symbol_list = tk.Listbox(
            bt_left, selectmode="browse",
            bg="#0f172a", fg="#e2e8f0",
            width=22, height=20
        )
        bt_list_v = ttk.Scrollbar(
            bt_left, orient="vertical",
            command=self.backtest_symbol_list.yview
        )
        self.backtest_symbol_list.configure(
            yscrollcommand=bt_list_v.set
        )
        self.backtest_symbol_list.pack(side="left", fill="y", expand=True)
        bt_list_v.pack(side="right", fill="y")

        for _s in SYMBOLS:
            self.backtest_symbol_list.insert("end", _s)

        def _pick_backtest_symbol(_event=None):
            try:
                sel = self.backtest_symbol_list.curselection()
                if sel:
                    value = self.backtest_symbol_list.get(sel[0])
                    self.backtest_symbol_combo.set(value)
            except Exception:
                pass

        self.backtest_symbol_list.bind("<<ListboxSelect>>", _pick_backtest_symbol)
        _install_scroll_support(self.backtest_symbol_list)

        bt_text_wrap = tk.Frame(bt_body)
        bt_text_wrap.pack(side="left", fill="both", expand=True)

        self.bt_text = tk.Text(
            bt_text_wrap, bg="#0f172a",
            fg="#e2e8f0", font=("Consolas", 10),
            wrap="none"
        )
        bt_v = ttk.Scrollbar(
            bt_text_wrap, orient="vertical",
            command=self.bt_text.yview
        )
        bt_h = ttk.Scrollbar(
            bt_text_wrap, orient="horizontal",
            command=self.bt_text.xview
        )
        self.bt_text.configure(
            yscrollcommand=bt_v.set, xscrollcommand=bt_h.set
        )
        self.bt_text.grid(row=0, column=0, sticky="nsew")
        bt_v.grid(row=0, column=1, sticky="ns")
        bt_h.grid(row=1, column=0, sticky="ew")
        bt_text_wrap.grid_rowconfigure(0, weight=1)
        bt_text_wrap.grid_columnconfigure(0, weight=1)
        _install_scroll_support(self.bt_text)
        self.bt_text.insert(
            "1.0",
            "یک نماد را از فهرست انتخاب کنید و «اجرای بک‌تست» را بزنید."
        )

        # --- NEW: Nobitex Authenticated API Tab ---
        tab_api = _make_scrollable_tab(self.notebook, bg="#0f172a")
        self.notebook.add(tab_api._scroll_outer, text="🟠 API نوبیتکس")
        api_top = tk.Frame(tab_api, bg="#1e293b")
        api_top.pack(fill="x", padx=8, pady=4)
        tk.Label(api_top, text="API اختصاصی نوبیتکس (معاملات واقعی)", fg="#0ea5e9", bg="#1e293b", font=("Tahoma", 12, "bold")).pack(side="left")
        tk.Label(api_top, text="⚠️ سفارش واقعی فقط با کلید REAL MARGIN در نوار بالا فعال می‌شود", fg="#fca5a5", bg="#1e293b", font=("Tahoma", 9, "bold")).pack(side="left", padx=20)

        # Token / API Key section
        token_frame = tk.LabelFrame(
            tab_api,
            text="احراز هویت نوبیتکس",
            bg="#1e293b",
            fg="white",
            font=("Tahoma", 10, "bold")
        )
        token_frame.pack(fill="x", padx=8, pady=4)

        tk.Label(token_frame, text="روش:", bg="#1e293b", fg="white").grid(
            row=0, column=0, padx=5, pady=4
        )
        self.api_auth_mode = ttk.Combobox(
            token_frame,
            values=["توکن قدیمی", "API Key جدید"],
            state="readonly",
            width=16
        )
        self.api_auth_mode.set(
            "API Key جدید" if globals().get("_nobitex_api_key", "") and globals().get("_nobitex_private_key", "")
            else "توکن قدیمی"
        )
        self.api_auth_mode.grid(row=0, column=1, padx=5, pady=4)

        tk.Label(token_frame, text="توکن API:", bg="#1e293b", fg="white").grid(
            row=1, column=0, padx=5, pady=4
        )
        self.api_token_entry = tk.Entry(token_frame, width=52, show="*")
        self.api_token_entry.grid(row=1, column=1, padx=5, pady=4, columnspan=3, sticky="ew")
        tk.Button(token_frame, text="کپی", command=lambda: _copy_entry(self.api_token_entry), bg="#334155", fg="white", width=7).grid(row=1, column=4, padx=2, pady=4)
        tk.Button(token_frame, text="پیست", command=lambda: _paste_entry(self.api_token_entry), bg="#2563eb", fg="white", width=7).grid(row=1, column=5, padx=2, pady=4)
        tk.Button(token_frame, text="پاک", command=lambda: _clear_entry(self.api_token_entry), bg="#7f1d1d", fg="white", width=7).grid(row=1, column=6, padx=2, pady=4)

        tk.Label(token_frame, text="Public Key:", bg="#1e293b", fg="white").grid(
            row=2, column=0, padx=5, pady=4
        )
        self.api_key_entry = tk.Entry(token_frame, width=52, show="*")
        self.api_key_entry.grid(row=2, column=1, padx=5, pady=4, columnspan=3, sticky="ew")
        tk.Button(token_frame, text="کپی", command=lambda: _copy_entry(self.api_key_entry), bg="#334155", fg="white", width=7).grid(row=2, column=4, padx=2, pady=4)
        tk.Button(token_frame, text="پیست", command=lambda: _paste_entry(self.api_key_entry), bg="#2563eb", fg="white", width=7).grid(row=2, column=5, padx=2, pady=4)
        tk.Button(token_frame, text="پاک", command=lambda: _clear_entry(self.api_key_entry), bg="#7f1d1d", fg="white", width=7).grid(row=2, column=6, padx=2, pady=4)

        tk.Label(token_frame, text="Private Key:", bg="#1e293b", fg="white").grid(
            row=3, column=0, padx=5, pady=4
        )
        self.api_private_key_entry = tk.Entry(token_frame, width=52, show="*")
        self.api_private_key_entry.grid(row=3, column=1, padx=5, pady=4, columnspan=3, sticky="ew")
        tk.Button(token_frame, text="کپی", command=lambda: _copy_entry(self.api_private_key_entry), bg="#334155", fg="white", width=7).grid(row=3, column=4, padx=2, pady=4)
        tk.Button(token_frame, text="پیست", command=lambda: _paste_entry(self.api_private_key_entry), bg="#2563eb", fg="white", width=7).grid(row=3, column=5, padx=2, pady=4)
        tk.Button(token_frame, text="پاک", command=lambda: _clear_entry(self.api_private_key_entry), bg="#7f1d1d", fg="white", width=7).grid(row=3, column=6, padx=2, pady=4)

        token_frame.grid_columnconfigure(1, weight=1)
        token_frame.grid_columnconfigure(2, weight=1)
        token_frame.grid_columnconfigure(3, weight=1)

        # Prefill saved values. The private key is only ever loaded into the local UI.
        try:
            self.api_token_entry.insert(0, globals().get("_nobitex_token", "") or "")
            self.api_key_entry.insert(0, globals().get("_nobitex_api_key", "") or "")
            self.api_private_key_entry.insert(0, globals().get("_nobitex_private_key", "") or "")
        except Exception:
            pass

        def _copy_entry(entry):
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(entry.get())
                self.root.update_idletasks()
            except Exception as e:
                logging.debug("API field copy failed: %s", e)

        def _paste_entry(entry):
            try:
                value = self.root.clipboard_get()
                entry.delete(0, tk.END)
                entry.insert(0, value)
            except Exception as e:
                logging.debug("API field paste failed: %s", e)

        def _clear_entry(entry):
            try:
                entry.delete(0, tk.END)
            except Exception:
                pass

        def save_api_credentials():
            token = self.api_token_entry.get().strip()
            api_key = self.api_key_entry.get().strip()
            private_key = self.api_private_key_entry.get().strip()

            mode = self.api_auth_mode.get().strip()
            global _nobitex_token, _nobitex_api_key, _nobitex_private_key

            if mode == "API Key جدید":
                if not api_key or not private_key:
                    messagebox.showwarning(
                        "خطا",
                        "برای API Key جدید، Public Key و Private Key را وارد کنید."
                    )
                    return
                _nobitex_api_key = api_key
                _nobitex_private_key = private_key
                # Keep the old token intact but do not use it in API-Key mode.
                _save_nobitex_config(
                    token=token,
                    api_key=api_key,
                    private_key=private_key,
                    mode="api_key"
                )
                messagebox.showinfo("موفق", "اطلاعات API Key ذخیره شد.")
            else:
                if not token:
                    messagebox.showwarning("خطا", "لطفاً توکن API را وارد کنید.")
                    return
                _nobitex_token = token
                _save_nobitex_config(
                    token=token,
                    api_key=api_key,
                    private_key=private_key,
                    mode="token"
                )
                messagebox.showinfo("موفق", "توکن API ذخیره شد.")

        def test_api_token():
            # The test always uses whichever authentication mode is selected.
            mode = self.api_auth_mode.get().strip()
            if mode == "API Key جدید":
                global _nobitex_api_key, _nobitex_private_key
                _nobitex_api_key = self.api_key_entry.get().strip()
                _nobitex_private_key = self.api_private_key_entry.get().strip()
                if not _nobitex_api_key or not _nobitex_private_key:
                    messagebox.showwarning(
                        "خطا",
                        "Public Key و Private Key را وارد کنید."
                    )
                    return
            else:
                global _nobitex_token
                _nobitex_token = self.api_token_entry.get().strip()
                if not _nobitex_token:
                    messagebox.showwarning("خطا", "توکن API خالی است.")
                    return

            ok, res = nobitex_get_profile()
            if ok:
                profile = res.get("profile", {}) if isinstance(res, dict) else {}
                name = (str(profile.get("firstName", "")).strip() + " " +
                        str(profile.get("lastName", "")).strip()).strip()
                messagebox.showinfo(
                    "اتصال موفق",
                    "احراز هویت نوبیتکس موفق بود.\n"
                    + (f"کاربر: {name}" if name else "اطلاعات پروفایل دریافت شد.")
                )
            else:
                messagebox.showerror("خطای احراز هویت", f"{res}")

        def login_api():
            dlg = tk.Toplevel(self.root)
            dlg.title("ورود به نوبیتکس")
            dlg.resizable(False, False)
            tk.Label(dlg, text="نام کاربری:").grid(row=0, column=0, padx=5, pady=5)
            user_entry = tk.Entry(dlg, width=30)
            user_entry.grid(row=0, column=1, padx=5, pady=5)
            tk.Label(dlg, text="رمز عبور:").grid(row=1, column=0, padx=5, pady=5)
            pass_entry = tk.Entry(dlg, width=30, show="*")
            pass_entry.grid(row=1, column=1, padx=5, pady=5)
            tk.Label(dlg, text="کد OTP (اختیاری):").grid(row=2, column=0, padx=5, pady=5)
            otp_entry = tk.Entry(dlg, width=30)
            otp_entry.grid(row=2, column=1, padx=5, pady=5)

            def do_login():
                ok, res = nobitex_login(
                    user_entry.get().strip(),
                    pass_entry.get(),
                    otp_entry.get().strip()
                )
                if ok:
                    self.api_auth_mode.set("توکن قدیمی")
                    self.api_token_entry.delete(0, tk.END)
                    self.api_token_entry.insert(0, res)
                    messagebox.showinfo("موفق", "ورود موفق و توکن دریافت شد.")
                    dlg.destroy()
                else:
                    messagebox.showerror("خطا", f"ورود ناموفق:\n{res}")

            tk.Button(
                dlg,
                text="ورود",
                command=do_login,
                bg=CTRL_BTN_BG,
                fg=CTRL_BTN_FG
            ).grid(row=3, column=0, columnspan=2, pady=10)

        tk.Button(
            token_frame, text="ذخیره اطلاعات",
            command=save_api_credentials,
            bg=CTRL_BTN_BG, fg=CTRL_BTN_FG
        ).grid(row=0, column=4, padx=5)

        tk.Button(
            token_frame, text="تست اتصال",
            command=test_api_token,
            bg="#10b981", fg="white"
        ).grid(row=0, column=5, padx=5)

        tk.Button(
            token_frame, text="ورود قدیمی",
            command=login_api,
            bg="#f59e0b", fg="black"
        ).grid(row=0, column=6, padx=5)

        tk.Label(
            token_frame,
            text="در API Key جدید: READ برای مشاهده و TRADE برای معامله لازم است.",
            bg="#1e293b", fg="#cbd5e1"
        ).grid(row=4, column=0, columnspan=7, padx=5, pady=3, sticky="w")

        api_clipboard_bar = tk.Frame(tab_api, bg="#111827")
        api_clipboard_bar.pack(fill="x", padx=8, pady=(0, 5))
        tk.Label(api_clipboard_bar, text="ابزار فیلد API:", bg="#111827", fg="#cbd5e1").pack(side="left", padx=6)
        tk.Button(api_clipboard_bar, text="کپی Token", command=lambda: _copy_entry(self.api_token_entry), bg="#334155", fg="white").pack(side="left", padx=2)
        tk.Button(api_clipboard_bar, text="پیست Token", command=lambda: _paste_entry(self.api_token_entry), bg="#2563eb", fg="white").pack(side="left", padx=2)
        tk.Button(api_clipboard_bar, text="کپی Public", command=lambda: _copy_entry(self.api_key_entry), bg="#334155", fg="white").pack(side="left", padx=2)
        tk.Button(api_clipboard_bar, text="پیست Public", command=lambda: _paste_entry(self.api_key_entry), bg="#2563eb", fg="white").pack(side="left", padx=2)
        tk.Button(api_clipboard_bar, text="کپی Private", command=lambda: _copy_entry(self.api_private_key_entry), bg="#334155", fg="white").pack(side="left", padx=2)
        tk.Button(api_clipboard_bar, text="پیست Private", command=lambda: _paste_entry(self.api_private_key_entry), bg="#2563eb", fg="white").pack(side="left", padx=2)

        # Wallets section
        wallets_frame = tk.LabelFrame(tab_api, text="کیف پول‌ها", bg="#1e293b", fg="white", font=("Tahoma", 10, "bold"))
        wallets_frame.pack(fill="both", expand=True, padx=8, pady=4)

        wallet_cols = ("wallet_type", "currency", "balance", "blocked", "available")
        wallet_titles = {"wallet_type": "نوع کیف پول", "currency": "ارز", "balance": "موجودی", "blocked": "مسدود", "available": "قابل استفاده"}
        wallet_table_wrap = tk.Frame(wallets_frame)
        wallet_table_wrap.pack(fill="both", expand=True, padx=5, pady=5)
        self.api_wallets_table = ttk.Treeview(wallet_table_wrap, columns=wallet_cols, show="headings")
        wallet_vsb = ttk.Scrollbar(wallet_table_wrap, orient="vertical", command=self.api_wallets_table.yview)
        wallet_hsb = ttk.Scrollbar(wallet_table_wrap, orient="horizontal", command=self.api_wallets_table.xview)
        self.api_wallets_table.configure(yscrollcommand=wallet_vsb.set, xscrollcommand=wallet_hsb.set)
        for c in wallet_cols:
            self.api_wallets_table.heading(c, text=wallet_titles.get(c, c))
            self.api_wallets_table.column(c, width=150, anchor="center")
        self.api_wallets_table.grid(row=0, column=0, sticky="nsew")
        wallet_vsb.grid(row=0, column=1, sticky="ns")
        wallet_hsb.grid(row=1, column=0, sticky="ew")
        wallet_table_wrap.grid_rowconfigure(0, weight=1)
        wallet_table_wrap.grid_columnconfigure(0, weight=1)
        _install_scroll_support(self.api_wallets_table)

        # Separate live summaries make the two account balances immediately
        # visible without mixing Spot and Margin funds.
        wallet_summary = tk.Frame(wallets_frame, bg="#0f172a")
        wallet_summary.pack(fill="x", padx=5, pady=(5, 0))
        self.api_spot_wallet_summary_var = tk.StringVar(value="اسپات | RLS: — | USDT: —")
        self.api_margin_wallet_summary_var = tk.StringVar(value="تعهدی | RLS: — | USDT: —")
        tk.Label(wallet_summary, textvariable=self.api_spot_wallet_summary_var,
                 bg="#064e3b", fg="#bbf7d0", font=("Tahoma", 10, "bold"),
                 padx=10, pady=4).pack(side="left", padx=5)
        tk.Label(wallet_summary, textvariable=self.api_margin_wallet_summary_var,
                 bg="#78350f", fg="#fef3c7", font=("Tahoma", 10, "bold"),
                 padx=10, pady=4).pack(side="left", padx=5)

        def _wallet_summary_text(wallets, label):
            if not isinstance(wallets, list):
                wallets = []
            balances = {"RLS": "0", "USDT": "0"}
            for w in wallets:
                cur = str(w.get("currency", "")).upper()
                if cur in balances:
                    balances[cur] = str(w.get("activeBalance", w.get("balance", "0")))
            return f"{label} | RLS: {balances['RLS']} | USDT: {balances['USDT']}"

        def refresh_wallets():
            spot_ok, spot_res = nobitex_get_wallets_by_type("spot")
            margin_ok, margin_res = nobitex_get_wallets_by_type("margin")
            if not spot_ok and not margin_ok:
                messagebox.showerror("خطا", f"دریافت موجودی اسپات و تعهدی ناموفق بود:\n{spot_res}\n{margin_res}")
                return

            self.api_wallets_table.delete(*self.api_wallets_table.get_children())
            spot_wallets = spot_res.get("wallets", []) if spot_ok and isinstance(spot_res, dict) else []
            margin_wallets = margin_res.get("wallets", []) if margin_ok and isinstance(margin_res, dict) else []

            self.api_spot_wallet_summary_var.set(
                _wallet_summary_text(spot_wallets, "اسپات") if spot_ok else "اسپات | RLS: خطا | USDT: خطا"
            )
            self.api_margin_wallet_summary_var.set(
                _wallet_summary_text(margin_wallets, "تعهدی") if margin_ok else "تعهدی | RLS: خطا | USDT: خطا"
            )

            for wallet_label, wallets in (("اسپات", spot_wallets), ("تعهدی", margin_wallets)):
                for w in wallets:
                    vals = (
                        wallet_label,
                        str(w.get("currency", "")).upper(),
                        w.get("balance", "0"),
                        w.get("blockedBalance", w.get("blocked", "0")),
                        w.get("activeBalance", w.get("balance", "0"))
                    )
                    self.api_wallets_table.insert("", "end", values=vals)

        tk.Button(wallets_frame, text="بروزرسانی موجودی اسپات + تعهدی", command=refresh_wallets, bg=CTRL_BTN_BG, fg=CTRL_BTN_FG).pack(pady=5)

        # Order section — horizontally/vertically scrollable so all controls remain visible.
        order_scroll = tk.Frame(tab_api, bg="#111827")
        order_scroll.pack(fill="x", padx=8, pady=4)
        order_canvas = tk.Canvas(order_scroll, bg="#111827", highlightthickness=0, height=105)
        order_vsb = ttk.Scrollbar(order_scroll, orient="vertical", command=order_canvas.yview)
        order_hsb = ttk.Scrollbar(order_scroll, orient="horizontal", command=order_canvas.xview)
        order_canvas.configure(yscrollcommand=order_vsb.set, xscrollcommand=order_hsb.set)
        order_canvas.grid(row=0, column=0, sticky="nsew")
        order_vsb.grid(row=0, column=1, sticky="ns")
        order_hsb.grid(row=1, column=0, sticky="ew")
        order_scroll.grid_columnconfigure(0, weight=1)
        order_frame = tk.LabelFrame(order_canvas, text="سفارش‌گذاری", bg="#1e293b", fg="white", font=("Tahoma", 10, "bold"))
        order_window = order_canvas.create_window((0, 0), window=order_frame, anchor="nw")
        def _sync_order_scroll(_event=None):
            order_canvas.configure(scrollregion=order_canvas.bbox("all"))
        order_frame.bind("<Configure>", _sync_order_scroll)
        order_canvas.bind("<Configure>", lambda e: order_canvas.itemconfigure(order_window, height=max(82, e.height - 2)))
        _install_scroll_support(order_canvas)

        tk.Label(order_frame, text="نماد:", bg="#1e293b", fg="white").grid(row=0, column=0, padx=5, pady=5)
        self.api_order_symbol = tk.Entry(order_frame, width=15)
        self.api_order_symbol.insert(0, "BTCIRT")
        self.api_order_symbol.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(order_frame, text="جهت:", bg="#1e293b", fg="white").grid(row=0, column=2, padx=5, pady=5)
        self.api_order_side = ttk.Combobox(order_frame, values=["buy", "sell"], width=10)
        self.api_order_side.set("buy")
        self.api_order_side.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(order_frame, text="قیمت:", bg="#1e293b", fg="white").grid(row=0, column=4, padx=5, pady=5)
        self.api_order_price = tk.Entry(order_frame, width=15)
        self.api_order_price.grid(row=0, column=5, padx=5, pady=5)

        tk.Label(order_frame, text="مقدار:", bg="#1e293b", fg="white").grid(row=0, column=6, padx=5, pady=5)
        self.api_order_amount = tk.Entry(order_frame, width=15)
        self.api_order_amount.grid(row=0, column=7, padx=5, pady=5)

        def place_api_order():
            sym = self.api_order_symbol.get().strip()
            side = self.api_order_side.get()
            price = self.api_order_price.get().strip()
            amount = self.api_order_amount.get().strip()
            if not all([sym, side, price, amount]):
                messagebox.showwarning("خطا", "لطفاً همه فیلدها را پر کنید")
                return
            ok, res = nobitex_add_order(sym, side, price, amount)
            if ok:
                order_id = res.get("order", {}).get("id", "N/A")
                messagebox.showinfo("موفق", f"سفارش ثبت شد!\nشناسه: {order_id}")
            else:
                messagebox.showerror("خطا", f"ثبت سفارش ناموفق: {res}")

        tk.Button(order_frame, text="ثبت سفارش", command=place_api_order, bg="#10b981", fg="white").grid(row=0, column=8, padx=10)

        # Withdraw section
        withdraw_frame = tk.LabelFrame(tab_api, text="برداشت", bg="#1e293b", fg="white", font=("Tahoma", 10, "bold"))
        withdraw_frame.pack(fill="x", padx=8, pady=4)

        tk.Label(withdraw_frame, text="ارز:", bg="#1e293b", fg="white").grid(row=0, column=0, padx=5, pady=5)
        self.api_withdraw_currency = tk.Entry(withdraw_frame, width=10)
        self.api_withdraw_currency.insert(0, "btc")
        self.api_withdraw_currency.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(withdraw_frame, text="آدرس:", bg="#1e293b", fg="white").grid(row=0, column=2, padx=5, pady=5)
        self.api_withdraw_address = tk.Entry(withdraw_frame, width=40)
        self.api_withdraw_address.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(withdraw_frame, text="مقدار:", bg="#1e293b", fg="white").grid(row=0, column=4, padx=5, pady=5)
        self.api_withdraw_amount = tk.Entry(withdraw_frame, width=15)
        self.api_withdraw_amount.grid(row=0, column=5, padx=5, pady=5)

        tk.Label(withdraw_frame, text="شبکه:", bg="#1e293b", fg="white").grid(row=0, column=6, padx=5, pady=5)
        self.api_withdraw_network = tk.Entry(withdraw_frame, width=10)
        self.api_withdraw_network.insert(0, "TRX")
        self.api_withdraw_network.grid(row=0, column=7, padx=5, pady=5)

        def do_withdraw():
            ok, res = nobitex_withdraw(
                self.api_withdraw_currency.get(),
                self.api_withdraw_address.get(),
                self.api_withdraw_amount.get(),
                self.api_withdraw_network.get()
            )
            if ok:
                messagebox.showinfo("موفق", f"درخواست برداشت ثبت شد!\n{res}")
            else:
                messagebox.showerror("خطا", f"برداشت ناموفق: {res}")

        tk.Button(withdraw_frame, text="درخواست برداشت", command=do_withdraw, bg="#ef4444", fg="white").grid(row=0, column=8, padx=10)

        # Deposit address section
        deposit_frame = tk.LabelFrame(tab_api, text="آدرس واریز", bg="#1e293b", fg="white", font=("Tahoma", 10, "bold"))
        deposit_frame.pack(fill="x", padx=8, pady=4)

        tk.Label(deposit_frame, text="ارز:", bg="#1e293b", fg="white").grid(row=0, column=0, padx=5, pady=5)
        self.api_deposit_currency = tk.Entry(deposit_frame, width=10)
        self.api_deposit_currency.insert(0, "btc")
        self.api_deposit_currency.grid(row=0, column=1, padx=5, pady=5)

        self.api_deposit_address_label = tk.Label(deposit_frame, text="آدرس: —", bg="#1e293b", fg="#0ea5e9", font=("Tahoma", 10, "bold"))
        self.api_deposit_address_label.grid(row=0, column=2, padx=10, pady=5)

        def get_deposit_addr():
            ok, res = nobitex_deposit_address(self.api_deposit_currency.get())
            if ok:
                addr = res.get("address", "N/A")
                self.api_deposit_address_label.config(text=f"آدرس: {addr}")
            else:
                messagebox.showerror("خطا", f"دریافت آدرس ناموفق: {res}")

        tk.Button(deposit_frame, text="دریافت آدرس", command=get_deposit_addr, bg=CTRL_BTN_BG, fg=CTRL_BTN_FG).grid(row=0, column=3, padx=10)

        # Orders list section
        orders_frame = tk.LabelFrame(tab_api, text="سفارش‌های باز", bg="#1e293b", fg="white", font=("Tahoma", 10, "bold"))
        orders_frame.pack(fill="both", expand=True, padx=8, pady=4)

        order_list_cols = ("id", "symbol", "side", "price", "amount", "status")
        order_list_titles = {"id": "شناسه", "symbol": "نماد", "side": "جهت", "price": "قیمت", "amount": "مقدار", "status": "وضعیت"}
        order_table_wrap = tk.Frame(orders_frame)
        order_table_wrap.pack(fill="both", expand=True, padx=5, pady=5)
        self.api_orders_table = ttk.Treeview(order_table_wrap, columns=order_list_cols, show="headings")
        order_vsb = ttk.Scrollbar(order_table_wrap, orient="vertical", command=self.api_orders_table.yview)
        order_hsb = ttk.Scrollbar(order_table_wrap, orient="horizontal", command=self.api_orders_table.xview)
        self.api_orders_table.configure(yscrollcommand=order_vsb.set, xscrollcommand=order_hsb.set)
        for c in order_list_cols:
            self.api_orders_table.heading(c, text=order_list_titles.get(c, c))
            self.api_orders_table.column(c, width=120, anchor="center")
        self.api_orders_table.grid(row=0, column=0, sticky="nsew")
        order_vsb.grid(row=0, column=1, sticky="ns")
        order_hsb.grid(row=1, column=0, sticky="ew")
        order_table_wrap.grid_rowconfigure(0, weight=1)
        order_table_wrap.grid_columnconfigure(0, weight=1)
        _install_scroll_support(self.api_orders_table)

        def refresh_orders():
            ok, res = nobitex_get_orders("open")
            if ok:
                self.api_orders_table.delete(*self.api_orders_table.get_children())
                orders = res.get("orders", [])
                for o in orders:
                    vals = (
                        o.get("id", ""),
                        o.get("srcCurrency", "").upper() + ("IRT" if o.get("dstCurrency") == "rls" else "USDT"),
                        o.get("type", ""),
                        o.get("price", ""),
                        o.get("amount", ""),
                        o.get("status", "")
                    )
                    self.api_orders_table.insert("", "end", values=vals)
            else:
                messagebox.showerror("خطا", f"دریافت سفارش‌ها ناموفق: {res}")

        def cancel_selected_order():
            sel = self.api_orders_table.selection()
            if not sel:
                messagebox.showwarning("خطا", "یک سفارش انتخاب کنید")
                return
            order_id = self.api_orders_table.item(sel[0])["values"][0]
            ok, res = nobitex_cancel_order(str(order_id))
            if ok:
                messagebox.showinfo("موفق", "سفارش لغو شد!")
                refresh_orders()
            else:
                messagebox.showerror("خطا", f"لغو ناموفق: {res}")

        btn_frame = tk.Frame(orders_frame, bg="#1e293b")
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="بروزرسانی سفارش‌ها", command=refresh_orders, bg=CTRL_BTN_BG, fg=CTRL_BTN_FG).pack(side="left", padx=5)
        tk.Button(btn_frame, text="لغو سفارش انتخابی", command=cancel_selected_order, bg="#ef4444", fg="white").pack(side="left", padx=5)


        # --- Adaptive Research / Optimization Tab ---
        # Learning/optimization is integrated into the Unified Market Intelligence Center.
        # --- UI Layout / Settings Tab ---
        self._build_layout_settings_tab()
        self._apply_saved_layout_settings()

    def _build_layout_settings_tab(self):
        """Personalize tab visibility, table columns and column widths; persist to disk."""
        try:
            self._layout_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_layout_settings.json")
            self._layout_tabs = {}
            self._layout_tables = {}
            self._layout_column_vars = {}
            self._layout_width_vars = {}

            for idx, tab_id in enumerate(self.notebook.tabs()):
                try:
                    label = self.notebook.tab(tab_id, "text")
                    widget = self.notebook.nametowidget(tab_id)
                    self._layout_tabs[str(tab_id)] = {"widget": widget, "text": label, "index": idx}
                except Exception:
                    pass

            table_specs = [
                ("main_table", "نمای کلی"), ("signals_table", "سیگنال‌ها"),
                ("autotrade_table", "اتو ترید"), ("deep_table", "آزمایشگاه تحلیل عمیق"),
                ("ai_table", "AI Trader"), ("smart_table", "اجرای هوشمند"),
                ("risk_table", "ریسک و سرمایه"), ("api_wallets_table", "کیف پول‌های API"),
                ("api_orders_table", "سفارش‌های API"), ("signals_v2_table", "سیگنال‌های نسخه ۲"),
            ]
            for attr, title in table_specs:
                tree = getattr(self, attr, None)
                if tree is not None:
                    self._layout_tables[attr] = {"widget": tree, "title": title}

            tab = _make_scrollable_tab(self.notebook, bg="#0f172a")
            self._layout_settings_tab = tab
            self.notebook.add(tab._scroll_outer, text="🔷 تنظیمات نمایش")

            top = tk.Frame(tab, bg="#1e293b")
            top.pack(fill="x", padx=8, pady=8)
            tk.Label(top, text="تنظیمات نمایش همه تب‌ها و ستون‌ها", bg="#1e293b", fg="white",
                     font=("Tahoma", 13, "bold")).pack(side="left", padx=10)
            tk.Button(top, text="ذخیره", command=self._save_layout_settings, bg="#10b981", fg="white", width=10).pack(side="right", padx=4)
            tk.Button(top, text="بازنشانی همه", command=self._reset_layout_settings, bg="#ef4444", fg="white", width=12).pack(side="right", padx=4)

            body = tk.Frame(tab, bg="#0f172a")
            body.pack(fill="both", expand=True, padx=8, pady=4)

            tabs_box = tk.LabelFrame(body, text="تب‌ها / منوهای اصلی", bg="#111827", fg="#93c5fd",
                                     font=("Tahoma", 10, "bold"), padx=8, pady=8)
            tabs_box.pack(fill="x", pady=(0, 8))
            self._layout_tab_vars = {}
            for key, info in self._layout_tabs.items():
                v = tk.BooleanVar(value=True)
                self._layout_tab_vars[key] = v
                tk.Checkbutton(tabs_box, text=info["text"], variable=v, command=self._apply_tab_visibility,
                               bg="#111827", fg="white", selectcolor="#1e293b", activebackground="#111827",
                               activeforeground="white").pack(side="left", padx=8, pady=4)

            tables_canvas = tk.Canvas(body, bg="#0f172a", highlightthickness=0)
            scrollbar = ttk.Scrollbar(body, orient="vertical", command=tables_canvas.yview)
            tables_inner = tk.Frame(tables_canvas, bg="#0f172a")
            tables_inner.bind("<Configure>", lambda e: tables_canvas.configure(scrollregion=tables_canvas.bbox("all")))
            tables_canvas.create_window((0,0), window=tables_inner, anchor="nw")
            tables_canvas.configure(yscrollcommand=scrollbar.set)
            tables_canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            for attr, info in self._layout_tables.items():
                tree = info["widget"]
                box = tk.LabelFrame(tables_inner, text=info["title"], bg="#111827", fg="#93c5fd",
                                    font=("Tahoma", 10, "bold"), padx=8, pady=6)
                box.pack(fill="x", padx=4, pady=5)
                tk.Label(box, text="نمایش ستون‌ها و عرض هر ستون:", bg="#111827", fg="#cbd5e1",
                         font=("Tahoma", 9, "bold")).pack(anchor="w")
                cols_frame = tk.Frame(box, bg="#111827")
                cols_frame.pack(fill="x", pady=3)
                for i, col in enumerate(tree["columns"]):
                    title = tree.heading(col, "text") or col
                    var = tk.BooleanVar(value=True)
                    self._layout_column_vars[(attr, col)] = var
                    cb = tk.Checkbutton(cols_frame, text=title, variable=var,
                                        command=lambda a=attr: self._apply_table_layout(a),
                                        bg="#111827", fg="white", selectcolor="#1e293b",
                                        activebackground="#111827", activeforeground="white", anchor="w")
                    cb.grid(row=i//4, column=i%4, sticky="w", padx=5, pady=2)
                    width_var = tk.IntVar(value=int(tree.column(col, "width") or 110))
                    self._layout_width_vars[(attr, col)] = width_var
                    spin = tk.Spinbox(cols_frame, from_=50, to=500, increment=10, width=5,
                                      textvariable=width_var, command=lambda a=attr: self._apply_table_layout(a),
                                      bg="#0b1220", fg="white", insertbackground="white")
                    spin.grid(row=i//4, column=i%4, sticky="e", padx=(0,8), pady=2)
                    spin.bind("<Return>", lambda e, a=attr: self._apply_table_layout(a))
                tk.Button(box, text="نمایش همه ستون‌ها", command=lambda a=attr: self._show_all_table_columns(a),
                          bg="#2563eb", fg="white").pack(side="left", padx=4, pady=4)
                tk.Button(box, text="فقط ستون اول", command=lambda a=attr: self._hide_all_but_first(a),
                          bg="#475569", fg="white").pack(side="left", padx=4, pady=4)

            self._layout_info_label = tk.Label(tab,
                text="تغییرات ستون‌ها و تب‌ها بلافاصله اعمال می‌شوند و در فایل ui_layout_settings.json ذخیره می‌شوند.",
                bg="#0f172a", fg="#94a3b8", font=("Tahoma", 9))
            self._layout_info_label.pack(fill="x", padx=10, pady=5)
        except Exception as exc:
            logging.exception("build layout settings failed: %s", exc)

    def _apply_tab_visibility(self):
        try:
            visible_widgets = [self.notebook.nametowidget(x) for x in self.notebook.tabs()]
            for key, info in self._layout_tabs.items():
                widget = info["widget"]
                want = bool(self._layout_tab_vars[key].get())
                visible = widget in visible_widgets
                if want and not visible:
                    idx = min(info.get("index", self.notebook.index("end")), self.notebook.index("end"))
                    self.notebook.insert(idx, widget)
                    visible_widgets = [self.notebook.nametowidget(x) for x in self.notebook.tabs()]
                elif not want and visible:
                    self.notebook.forget(widget)
                    visible_widgets = [self.notebook.nametowidget(x) for x in self.notebook.tabs()]
            self._save_layout_settings()
        except Exception as exc:
            logging.debug("apply tab visibility failed: %s", exc)

    def _apply_table_layout(self, attr):
        try:
            info = self._layout_tables.get(attr)
            if not info: return
            tree = info["widget"]
            cols = list(tree["columns"])
            visible = [c for c in cols if self._layout_column_vars[(attr, c)].get()]
            if not visible and cols:
                visible = [cols[0]]
                self._layout_column_vars[(attr, cols[0])].set(True)
            tree.configure(displaycolumns=visible)
            for c in cols:
                try:
                    tree.column(c, width=max(50, min(500, int(self._layout_width_vars[(attr,c)].get()))))
                except Exception:
                    pass
        except Exception as exc:
            logging.debug("apply table layout failed %s: %s", attr, exc)

    def _show_all_table_columns(self, attr):
        for (a, c), v in self._layout_column_vars.items():
            if a == attr: v.set(True)
        self._apply_table_layout(attr)
        self._save_layout_settings()

    def _hide_all_but_first(self, attr):
        info = self._layout_tables.get(attr)
        if not info: return
        cols = list(info["widget"]["columns"])
        for c in cols: self._layout_column_vars[(attr,c)].set(c == cols[0])
        self._apply_table_layout(attr)
        self._save_layout_settings()

    def _save_layout_settings(self):
        try:
            import json
            data = {"tabs": {}, "tables": {}}
            for key in self._layout_tabs:
                data["tabs"][key] = bool(self._layout_tab_vars[key].get())
            for attr, info in self._layout_tables.items():
                tree = info["widget"]
                data["tables"][attr] = {
                    "columns": {c: bool(self._layout_column_vars[(attr,c)].get()) for c in tree["columns"]},
                    "widths": {c: int(self._layout_width_vars[(attr,c)].get()) for c in tree["columns"]}
                }
            with open(self._layout_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if hasattr(self, "_layout_info_label"):
                self._layout_info_label.config(text="✓ تنظیمات نمایش ذخیره شد.")
        except Exception as exc:
            logging.debug("save layout settings failed: %s", exc)

    def _apply_saved_layout_settings(self):
        try:
            import json
            if os.path.exists(self._layout_file):
                with open(self._layout_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key, val in data.get("tabs", {}).items():
                    if key in self._layout_tab_vars: self._layout_tab_vars[key].set(bool(val))
                for attr, cfg in data.get("tables", {}).items():
                    if attr not in self._layout_tables: continue
                    for c, val in cfg.get("columns", {}).items():
                        if (attr,c) in self._layout_column_vars: self._layout_column_vars[(attr,c)].set(bool(val))
                    for c, val in cfg.get("widths", {}).items():
                        if (attr,c) in self._layout_width_vars:
                            try: self._layout_width_vars[(attr,c)].set(int(val))
                            except Exception: pass
            for attr in self._layout_tables: self._apply_table_layout(attr)
            self._apply_tab_visibility()
        except Exception as exc:
            logging.debug("apply saved layout failed: %s", exc)

    def _reset_layout_settings(self):
        try:
            for v in self._layout_tab_vars.values(): v.set(True)
            for (attr,c), v in self._layout_column_vars.items():
                v.set(True)
                try: self._layout_width_vars[(attr,c)].set(110)
                except Exception: pass
            for attr in self._layout_tables: self._apply_table_layout(attr)
            self._apply_tab_visibility()
            self._save_layout_settings()
        except Exception as exc:
            logging.debug("reset layout settings failed: %s", exc)

    def _build_adaptive_research_tab(self):
        """Unified research/learning center: history -> optimization -> OOS ML -> calibration -> deployment."""
        try:
            tab = tk.Frame(self.notebook, bg="#0f172a")
            self.unified_learning_tab = tab
            # MERGED INTO UNIFIED MARKET INTELLIGENCE CENTER: no standalone tab.

            top = tk.Frame(tab, bg="#1e293b")
            top.pack(fill="x", padx=8, pady=8)
            tk.Label(top, text="مرکز جامع یادگیری، بک‌تست، کشف بهترین شرایط سیگنال و استقرار خودکار",
                     bg="#1e293b", fg="#f8fafc", font=("Tahoma", 13, "bold")).pack(side="left", padx=8)

            controls = tk.Frame(tab, bg="#111827")
            controls.pack(fill="x", padx=8, pady=(0, 6))
            tk.Label(controls, text="دوره آموزش:", bg="#111827", fg="white").pack(side="left", padx=(8,3))
            self.unified_years_var = tk.StringVar(value="2")
            ttk.Combobox(controls, textvariable=self.unified_years_var, values=["1","2","3","5"], width=5, state="readonly").pack(side="left", padx=3)
            tk.Label(controls, text="دامنه:", bg="#111827", fg="white").pack(side="left", padx=(12,3))
            self.unified_scope_var = tk.StringVar(value="همه نمادها")
            ttk.Combobox(controls, textvariable=self.unified_scope_var, values=["همه نمادها","نماد انتخابی"], width=13, state="readonly").pack(side="left", padx=3)
            self.unified_symbol_var = tk.StringVar(value=(SYMBOLS[0] if SYMBOLS else ""))
            self.unified_symbol_combo = ttk.Combobox(controls, textvariable=self.unified_symbol_var, values=list(SYMBOLS), width=16, state="readonly")
            self.unified_symbol_combo.pack(side="left", padx=3)
            tk.Label(controls, text="تایم‌فریم:", bg="#111827", fg="white").pack(side="left", padx=(12,3))
            self.unified_tf_var = tk.StringVar(value="همه")
            ttk.Combobox(controls, textvariable=self.unified_tf_var,
                          values=["همه","1m","5m","15m","30m","1h","4h","1d"], width=8, state="readonly").pack(side="left", padx=3)

            self.unified_progress = tk.Label(controls, text="آماده", bg="#111827", fg="#fbbf24", font=("Tahoma",9,"bold"))
            self.unified_progress.pack(side="left", padx=12)
            tk.Button(controls, text="▶ اجرای کامل Pipeline", command=self._run_unified_learning_pipeline,
                      bg="#059669", fg="white", font=("Tahoma",9,"bold"), padx=10).pack(side="right", padx=4)
            tk.Button(controls, text="فقط ML Walk-Forward", command=lambda:self._run_unified_learning_pipeline(ml_only=True),
                      bg="#2563eb", fg="white").pack(side="right", padx=4)
            tk.Button(controls, text="گزارش Champion", command=self._unified_show_report,
                      bg="#7c3aed", fg="white").pack(side="right", padx=4)
            tk.Button(controls, text="بک‌تست کلاسیک", command=self.run_backtest_dialog,
                      bg="#475569", fg="white").pack(side="right", padx=4)

            stage = tk.Frame(tab, bg="#0f172a")
            stage.pack(fill="x", padx=8, pady=5)
            self.unified_stage_vars = {}
            for key, title in [
                ("history","۱. تاریخچه"),("optimizer","۲. بهینه‌سازی"),("ml","۳. ML / OOS"),
                ("calibration","۴. کالیبراسیون"),("champion","۵. Champion"),("deploy","۶. استفاده در سیگنال")]:
                box=tk.Frame(stage,bg="#111827",bd=1,relief="solid")
                box.pack(side="left",fill="x",expand=True,padx=2)
                tk.Label(box,text=title,bg="#111827",fg="#cbd5e1",font=("Tahoma",8,"bold")).pack(pady=(5,1))
                v=tk.Label(box,text="○ آماده",bg="#111827",fg="#94a3b8",font=("Tahoma",9,"bold"))
                v.pack(pady=(0,5)); self.unified_stage_vars[key]=v

            summary = tk.LabelFrame(tab, text="خلاصه وضعیت یادگیری", bg="#0f172a", fg="#93c5fd", font=("Tahoma",10,"bold"))
            summary.pack(fill="x", padx=8, pady=5)
            self.unified_summary = tk.Text(summary, height=7, bg="#0b1220", fg="#e5e7eb", font=("Consolas",9), wrap="word")
            self.unified_summary.pack(fill="x", padx=6, pady=5)

            body = tk.PanedWindow(tab, orient="vertical", bg="#0f172a", sashwidth=6)
            body.pack(fill="both", expand=True, padx=8, pady=5)
            report_box=tk.LabelFrame(body,text="نتایج قابل استفاده توسط موتور سیگنال",bg="#0f172a",fg="#22d3ee")
            self.unified_results=tk.Text(report_box,bg="#0b1220",fg="#e5e7eb",font=("Consolas",9),wrap="none")
            urv=ttk.Scrollbar(report_box,orient="vertical",command=self.unified_results.yview); urh=ttk.Scrollbar(report_box,orient="horizontal",command=self.unified_results.xview)
            self.unified_results.configure(yscrollcommand=urv.set,xscrollcommand=urh.set)
            self.unified_results.grid(row=0,column=0,sticky="nsew"); urv.grid(row=0,column=1,sticky="ns"); urh.grid(row=1,column=0,sticky="ew")
            report_box.grid_rowconfigure(0,weight=1); report_box.grid_columnconfigure(0,weight=1); body.add(report_box,stretch="always")
            log_box=tk.LabelFrame(body,text="گزارش اجرای Pipeline",bg="#0f172a",fg="#fbbf24")
            self.unified_log=tk.Text(log_box,height=8,bg="#020617",fg="#cbd5e1",font=("Consolas",9),wrap="word")
            self.unified_log.pack(fill="both",expand=True,padx=5,pady=5); body.add(log_box,stretch="always")
            _install_scroll_support(self.unified_results); _install_scroll_support(self.unified_log); _install_scroll_support(self.unified_summary)
            self._refresh_unified_learning_ui()
        except Exception as exc:
            logging.exception("build unified learning tab failed: %s",exc)

    def _unified_tfs(self):
        m={"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}
        v=getattr(self,"unified_tf_var",None).get() if hasattr(self,"unified_tf_var") else "همه"
        return list(m.values()) if v=="همه" else [m.get(v,15)]

    def _unified_symbols(self):
        if getattr(self,"unified_scope_var",None).get()=="نماد انتخابی":
            return [canonical(self.unified_symbol_var.get())]
        return list(SYMBOLS)

    def _unified_stage(self,key,state="done",text=None):
        if not hasattr(self,"unified_stage_vars") or key not in self.unified_stage_vars: return
        icon="✓" if state=="done" else "●" if state=="active" else "!" if state=="error" else "○"
        color="#22c55e" if state=="done" else "#fbbf24" if state=="active" else "#ef4444" if state=="error" else "#94a3b8"
        label=f"{icon} {text or ('انجام شد' if state=='done' else 'در حال اجرا' if state=='active' else 'آماده')}"
        def apply():
            try: self.unified_stage_vars[key].config(text=label,fg=color)
            except Exception: pass
        try: self.root.after(0,apply)
        except Exception: apply()

    def _unified_log(self,msg):
        try:
            self.root.after(0,lambda:self.unified_log.insert("end",f"[{time.strftime('%H:%M:%S')}] {msg}\n"))
            self.root.after(0,lambda:self.unified_log.see("end"))
        except Exception: pass

    def _run_unified_learning_pipeline(self, ml_only=False):
        if getattr(self,"_unified_pipeline_running",False): return
        self._unified_pipeline_running=True
        years=int(self.unified_years_var.get() or 2); syms=self._unified_symbols(); tfs=self._unified_tfs()
        def worker():
            try:
                self.root.after(0,lambda:self.unified_progress.config(text="در حال اجرا..."))
                self._unified_log(f"شروع Pipeline | symbols={len(syms)} | TF={tfs} | years={years} | ML-only={ml_only}")
                self._unified_stage("history","active","جمع‌آوری/تکمیل")
                # adaptive_research_all is resumable and stores candles in SQLite; when ML-only,
                # it reuses the existing stored history.
                if not ml_only:
                    self._unified_stage("optimizer","active","در حال بهینه‌سازی")
                if not ml_only:
                    # Run the existing integrated research engine and wait for its worker to finish.
                    adaptive_research_all(syms,tfs,years=years,train_ml=True)
                    while True:
                        with _adaptive_lock: running=bool(_adaptive_research_status.get("running"))
                        if not running: break
                        self.root.after(0,lambda:self.unified_progress.config(text=f"تحقیق: {_adaptive_research_status.get('progress',0)}%"))
                        time.sleep(1.0)
                    self._unified_stage("history","done","ممیزی کندل‌ها تکمیل شد")
                    self._unified_log("ممیزی دقیق تاریخچه انجام شد: برای هر نماد/TF تعداد مورد انتظار، تعداد واقعی، گپ‌ها و پوشش زمانی ثبت شد؛ داده ناقص وارد آموزش نشد.")
                    self._unified_stage("optimizer","done","بهترین پارامترها ذخیره شد")
                else:
                    self._unified_stage("history","done","استفاده از تاریخچه ذخیره‌شده")
                self._unified_stage("ml","active","Walk-Forward OOS")
                total=max(1,len(syms)*len(tfs)); done=0
                for sym in syms:
                    for tf in tfs:
                        try:
                            audit=adaptive_history_coverage(sym,tf,years)
                            adaptive_save_history_audit(audit)
                            if audit.get("complete"):
                                df=adaptive_load_candles(sym,tf,audit["start"],audit["end"])
                                if len(df)>=ADAPTIVE_MIN_TRAIN_ROWS:
                                    adaptive_train_ml_champions(sym,tf,df)
                            else:
                                self._unified_log(f"ML رد شد؛ تاریخچه ناقص {sym}/{tf}m: {audit.get('count',0)}/{audit.get('expected',0)}")
                        except Exception as exc: self._unified_log(f"ML {sym}/{tf}: {exc}")
                        done+=1
                        self.root.after(0,lambda d=done,t=total:self.unified_progress.config(text=f"ML: {d}/{t}"))
                self._unified_stage("ml","done","OOS انجام شد")
                self._unified_stage("calibration","done","Probability Calibration")
                self._unified_stage("champion","active","انتخاب مدل/پارامتر")
                self._unified_show_report()
                self._unified_stage("champion","done","Champion ذخیره شد")
                self._unified_stage("deploy","done","موتور سیگنال از نتایج استفاده می‌کند")
                self._unified_log("پایان موفق: نتایج در SQLite و artifactها ذخیره شدند و adaptive_enrich_signal از آنها استفاده می‌کند.")
                self.root.after(0,lambda:self.unified_progress.config(text="✓ آماده استفاده"))
            except Exception as exc:
                logging.exception("unified learning pipeline failed")
                self._unified_log(f"خطای Pipeline: {exc}")
                self._unified_stage("champion","error","خطا")
                self.root.after(0,lambda:self.unified_progress.config(text="خطا؛ گزارش را بررسی کنید"))
            finally: self._unified_pipeline_running=False
        threading.Thread(target=worker,daemon=True,name="UnifiedLearningPipeline").start()

    def _unified_show_report(self):
        try:
            adaptive_init_db()
            with adaptive_db() as db:
                opts=db.execute("""SELECT symbol,timeframe,direction,score,trades,win_rate,profit_factor,expectancy,max_drawdown,params_json FROM optimizer_results ORDER BY score DESC LIMIT 60""").fetchall()
                models=db.execute("""SELECT symbol,timeframe,model,samples,accuracy,auc,f1,brier,is_champion FROM model_results WHERE is_champion=1 ORDER BY auc DESC LIMIT 60""").fetchall()
                runs=db.execute("SELECT id,started_at,finished_at,status,message FROM research_runs ORDER BY id DESC LIMIT 5").fetchall()
                audits=db.execute("""SELECT symbol,timeframe,requested_years,expected_candles,actual_candles,missing_candles,coverage_pct,gap_count,largest_gap_candles,complete
                                      FROM history_audits ORDER BY checked_at DESC LIMIT 200""").fetchall()
            lines=["=== ممیزی دقیق تاریخچه کندل ===","نماد | TF | سال درخواستی | مورد انتظار | دریافت‌شده | مفقود | پوشش | گپ | بزرگترین گپ | وضعیت"]
            for r in audits:
                lines.append(f"{r[0]} | {r[1]}m | {r[2]:g} | {r[3]} | {r[4]} | {r[5]} | {r[6]:.2f}% | {r[7]} | {r[8]} | {'COMPLETE' if r[9] else 'INCOMPLETE'}")
            lines += ["","=== مرکز یادگیری: بهترین تنظیمات OOS ===","نماد | TF | جهت | Score | معاملات | Win% | PF | Expectancy | DD% | پارامترها"]
            for r in opts:
                try: p=json.loads(r[9]); ps=f"RSI={p.get('rsi_period')}/{p.get('rsi_os')}/{p.get('rsi_ob')} MACD={p.get('macd_fast')}/{p.get('macd_slow')}/{p.get('macd_signal')}"
                except Exception: ps="—"
                lines.append(f"{r[0]} | {r[1]}m | {r[2]} | {r[3]:.2f} | {r[4]} | {r[5]:.1f}% | {r[6]:.2f} | {r[7]:.4f} | {r[8]:.2f} | {ps}")
            lines += ["","=== ML Champions / Walk-Forward OOS ===","نماد | TF | مدل | Samples | Accuracy | AUC | F1 | Brier | Champion"]
            for r in models: lines.append(f"{r[0]} | {r[1]}m | {r[2]} | {r[3]} | {r[4]:.1f}% | {r[5]:.3f} | {r[6]:.1f}% | {r[7]:.4f} | YES")
            lines += ["","=== آخرین اجراها ==="]
            for r in runs: lines.append(f"run#{r[0]} | {r[3]} | {r[4]}")
            self.unified_results.delete("1.0","end"); self.unified_results.insert("1.0","\n".join(lines))
            self.unified_summary.delete("1.0","end")
            complete_count=sum(1 for r in audits if r[9])
            incomplete_count=sum(1 for r in audits if not r[9])
            self.unified_summary.insert("1.0",f"ممیزی تاریخچه: {complete_count} مورد کامل | {incomplete_count} مورد ناقص\n"+adaptive_status_text()+"\n\n✓ این نتایج persistent هستند و پس از راه‌اندازی مجدد نیز از SQLite/artifactها بارگذاری می‌شوند.\n✓ موتور سیگنال در adaptive_enrich_signal ابتدا تنظیمات ذخیره‌شده و سپس Champion ML را وارد امتیاز نهایی می‌کند؛ سیگنال WAIT فقط با این لایه به معامله تبدیل نمی‌شود.")
        except Exception as exc: logging.debug("unified report failed: %s",exc)

    def _refresh_unified_learning_ui(self):
        try:
            # Refresh the live dashboard while the worker is running. The old
            # version refreshed only the text report, so a pipeline could run
            # for hours while the table/cards appeared frozen.
            if hasattr(self,"unified_table"):
                self._unified_refresh_dashboard()
            if hasattr(self,"unified_summary") and not getattr(self,"_unified_pipeline_running",False):
                self._unified_show_report()
            self.root.after(2000,self._refresh_unified_learning_ui)
        except Exception:
            try: self.root.after(3000,self._refresh_unified_learning_ui)
            except Exception: pass

    def _refresh_adaptive_ui(self):
        try:
            if hasattr(self,"adaptive_status"):
                self.adaptive_status.delete("1.0","end"); self.adaptive_status.insert("1.0",adaptive_status_text())
                self.root.after(3000,self._refresh_adaptive_ui)
        except Exception: pass

    def _adaptive_refresh_fng(self):
        try:
            adaptive_fear_greed(force=True); self._refresh_adaptive_ui()
        except Exception: pass

    def _adaptive_show_champions(self):
        try:
            adaptive_init_db()
            with adaptive_db() as db:
                rows=db.execute("""SELECT symbol,timeframe,direction,score,trades,wins,win_rate,profit_factor,max_drawdown,params_json
                    FROM optimizer_results ORDER BY score DESC LIMIT 100""").fetchall()
                models=db.execute("""SELECT symbol,timeframe,model,accuracy,auc,f1,is_champion
                    FROM model_results WHERE is_champion=1 ORDER BY auc DESC LIMIT 100""").fetchall()
            lines=["=== بهترین تنظیمات ذخیره‌شده ==="]
            for r in rows:
                p=json.loads(r[9])
                lines.append(f"{r[0]} | {r[1]}m | {r[2]} | score={r[3]:.2f} | trades={r[4]} | win={r[6]:.1f}% | PF={r[7]:.2f} | DD={r[8]:.2f}% | RSI={p['rsi_period']}/{p['rsi_os']}/{p['rsi_ob']} | MACD={p['macd_fast']}/{p['macd_slow']}/{p['macd_signal']}")
            lines += ["", "=== ML Champion ==="]
            for r in models:
                lines.append(f"{r[0]} | {r[1]}m | {r[2]} | accuracy={r[3]:.1f}% | AUC={r[4]:.3f} | F1={r[5]:.1f}%")
            self.adaptive_results.delete("1.0","end"); self.adaptive_results.insert("1.0","\n".join(lines))
        except Exception as exc:
            logging.debug("adaptive show champions failed: %s",exc)

    def _adaptive_train_visible(self):
        def w():
            syms=list(SYMBOLS)
            for sym in syms:
                for tf in ADAPTIVE_RESEARCH_TIMEFRAMES:
                    try:
                        df=adaptive_load_candles(sym,tf)
                        if len(df)>=ADAPTIVE_MIN_TRAIN_ROWS: adaptive_train_ml_champions(sym,tf,df)
                    except Exception: pass
        threading.Thread(target=w,daemon=True).start()


    def self_trailing_enabled(self) -> bool:
            try:
                return bool(self.trailing_enabled.get())
            except Exception:
                return True

    def refresh_overview_prices(self):
        if getattr(self, "_overview_bootstrapped", False):
            return
        self._overview_bootstrapped = True

        """Populate prices without overwriting completed indicator rows."""
        try:
            updates = []
            for sym in list(SYMBOLS):
                key = canonical(sym)
                price = valid_symbols_map.get(key)
                if price is None:
                    continue
                # Exactly 20 fields: symbol..rules. No internal Talaye/target
                # bookkeeping fields are allowed into the Treeview.
                updates.append((
                    key, format_price(price), "", "—", "—", "—", "—", "—",
                    "—", "—", "در حال تحلیل", "نگهداری", 0, "50.0%", "0.00%",
                    "—", "0%", "neutral", "—", ""
                ))

            # Insert only missing symbols; never replace a row that already
            # contains actual indicator values.
            existing = {}
            for iid in self.main_table.get_children():
                vals = self.main_table.item(iid).get("values", [])
                if vals:
                    existing[str(vals[0])] = (iid, vals)

            for row in updates:
                sym = str(row[0])
                if sym in existing:
                    iid, current = existing[sym]
                    try:
                        if len(current) > 3 and current[3] not in ("—", "", "N/A"):
                            continue
                        self.main_table.item(iid, values=row)
                    except Exception:
                        pass
                else:
                    try:
                        self.main_table.insert(
                            "", "end", values=row, tags=("neutral",)
                        )
                    except Exception:
                        pass
        except Exception as e:
            logging.debug("refresh_overview_prices error: %s", e)


    def initial_fill_symbols(self):
        try:
            syms = SYMBOLS if SYMBOLS else [canonical(s) for s in USER_SYMBOLS]
            existing = {self.main_table.item(i)["values"][0]: i for i in self.main_table.get_children()}
            for sym in syms:
                if sym in pruned_symbols:
                    continue
                if sym in existing:
                    continue
                vals = (sym, "—", "—", "—", "—", "—", "—", "—", "—", "—",
                        "در حال تحلیل", "نگهداری", 0, "0.0%", "—", "—", "0%",
                        "neutral", "—", "")
                try:
                    self.main_table.insert("", "end", values=vals, tags=("neutral",))
                except Exception:
                    continue
        except Exception:
            pass

    def _populate_worker(self):
        try:
            active_symbols = list(SYMBOLS)
            if not active_symbols:
                active_symbols = [canonical(s) for s in USER_SYMBOLS]
            for s in active_symbols:
                try:
                    self.executor.submit(self._analyze_and_publish, s)
                except Exception:
                    pass
        except Exception:
            pass

    def _analyze_and_publish(self, sym: str):
        try:
            r = self.analyze_symbol_with_rules(sym)
            if r:
                self.root.after(0, lambda rr=r: (self.update_main_table([rr]), self.refresh_tables()))
        except Exception:
            pass

    def _sync_threadsafe_flags(self):
        # Tkinter variables must only be read from the Tk main thread.
        try:
            self._ml_enabled_state = bool(self.ml_enabled.get())
            self._auto_trade_state = bool(self.auto_trade.get())
        except Exception:
            pass

    def _ml_retrain_worker(self):
        while True:
            try:
                time.sleep(ML_RETRAIN_INTERVAL)
                if self._ml_enabled_state and _ML_AVAILABLE.get("sklearn"):
                    logging.info("Auto-retraining ML models...")
                    ml_train_ensemble(SAFE_FALLBACK_SYMBOLS)
                    global _ml_models_loaded
                    _ml_models_loaded = False
                    load_ml_ensemble()
            except Exception:
                pass

    def train_ml_models(self):
        try:
            if not _ML_AVAILABLE.get("sklearn"):
                messagebox.showerror("خطا", "scikit-learn نصب نشده است")
                return
            threading.Thread(target=lambda: ml_train_ensemble(SAFE_FALLBACK_SYMBOLS), daemon=True).start()
            self.root.after(0, lambda: messagebox.showinfo("آموزش ML", "آموزش مدل‌ها در پس‌زمینه شروع شد..."))
        except Exception as e:
            messagebox.showerror("خطا", str(e))


    def _deep_report_text(self, r):
        lines=[
            "="*88,
            f"گزارش تحلیل عمیق — {r.get('symbol','')}",
            "="*88,
            "دوره: حدود ۲ سال",
            f"امتیاز نهایی: {r.get('score',0):.2f}/100",
            f"بهترین تایم‌فریم: {r.get('best_tf','—')}",
            "",
            "سه روش اختصاصی:",
            "۱) تحلیل رژیم بازار: روند + مومنتوم + نوسان",
            "۲) فشار نقدینگی: موقعیت بسته‌شدن کندل + حجم نسبی",
            "۳) خمیدگی حرکت: شتاب و تغییر شیب مومنتوم",
            ""
        ]
        names={5:"۵ دقیقه",15:"۱۵ دقیقه",60:"۱ ساعت",240:"۴ ساعت",1440:"روزانه"}
        for tf in sorted(r.get("timeframes",{})):
            m=r["timeframes"][tf]
            lines += [
                f"━━━ {names.get(tf,str(tf))} ━━━",
                f"کندل: {m['bars']}",
                f"RSI: {m['rsi']:.2f} | MACD: {m['macd']:.6f} | ADX: {m['adx']:.2f} | DeMarker: {m['dem']:.3f}",
                f"بازده ۳۰ کندل: {m['ret30']:+.2f}% | بازده ۹۰ کندل: {m['ret90']:+.2f}%",
                f"نسبت حجم: {m['vol_ratio']:.2f}x",
                f"امتیاز روند: {m['trend']:.1f} | مومنتوم: {m['momentum']:.1f} | شاخص‌ها: {m['ind']:.1f}",
                f"روش ۱: {m['regime']:.1f} | روش ۲: {m['liquidity']:.1f} | روش ۳: {m['curvature']:.1f}",
                "الگوها: "+("، ".join(m['patterns']) if m['patterns'] else "—"),
                "حمایت: "+(" | ".join(format_price(x) for x in m['support']) if m['support'] else "—"),
                "مقاومت: "+(" | ".join(format_price(x) for x in m['resistance']) if m['resistance'] else "—"),
                f"امتیاز ترکیبی: {m['score']:.2f}",
                "━━━ تحقیق مستقل روش‌ها (Train 70% / OOS 30%) ━━━",
            ]
            mr=m.get("method_research",{})
            for mn,mr0 in mr.items():
                if mn=="COMBINATION": continue
                b=mr0.get("best",{}) if isinstance(mr0,dict) else {}
                lines.append(f"{mn}: OOS score={b.get('score',0):.1f} | WR={b.get('win_rate',0):.1f}% | PF={b.get('profit_factor',0):.2f} | DD={b.get('max_drawdown',0):.1f}% | params={b.get('params',{})}")
            cb=mr.get("COMBINATION",{}) if isinstance(mr,dict) else {}
            lines.append(f"ترکیب بهترین روش‌ها: score={cb.get('score',0):.1f} | WR={cb.get('win_rate',0):.1f}% | PF={cb.get('profit_factor',0):.2f} | DD={cb.get('max_drawdown',0):.1f}%")
            lines.append("")
        s=r.get("score",50)
        lines.append("━━━ نتیجه ━━━")
        lines.append("تمایل غالب: "+("صعودی" if s>=70 else "نزولی" if s<=30 else "خنثی"))
        lines.append("توجه: این موتور برای رتبه‌بندی و تحقیق است و تضمین سود نمی‌دهد.")
        return "\n".join(lines)

    def _deep_insert_result(self,r):
        per=r.get("timeframes",{})
        vals=[]
        for tf in (5,15,60,240,1440):
            vals.append(f"{per[tf]['score']:.1f}" if tf in per else "—")
        s=r.get("score",0)
        rec="خرید" if s>=70 else "فروش" if s<=30 else "انتظار"
        self.deep_table.insert("", "end",
            values=(r.get("symbol",""),f"{s:.1f}",str(r.get("best_tf","—")),
                    vals[0],vals[1],vals[2],vals[3],vals[4],rec))

    def run_deep_selected(self):
        sym=self.deep_symbol_combo.get().strip()
        if not sym:
            messagebox.showwarning("تحلیل عمیق","یک نماد انتخاب کنید.")
            return
        self.deep_table.delete(*self.deep_table.get_children())
        self.deep_report.delete("1.0","end")
        def worker():
            try:
                self.root.after(0,lambda:self.deep_progress.config(text=f"درحال تحلیل {sym}"))
                r=deep_analyze_symbol(sym)
                def done():
                    self._deep_insert_result(r)
                    self.deep_report.insert("1.0",self._deep_report_text(r))
                    self.deep_progress.config(text="پایان")
                self.root.after(0,done)
            except Exception as e:
                self.root.after(0,lambda err=str(e):self.deep_progress.config(text=f"خطا: {err}"))
        threading.Thread(target=worker,daemon=True).start()

    def run_deep_all(self):
        syms=[s for s in SYMBOLS if s not in pruned_symbols]
        if not syms:
            messagebox.showwarning("تحلیل عمیق","نمادی موجود نیست.")
            return
        if not messagebox.askyesno("تحلیل عمیق",
            f"تحلیل {len(syms)} نماد × {len(DEEP_TIMEFRAMES)} تایم‌فریم × حدود ۲ سال، زمان‌بر است. ادامه؟"):
            return
        self.deep_table.delete(*self.deep_table.get_children())
        self.deep_report.delete("1.0","end")
        def worker():
            results=[]
            for i,sym in enumerate(syms,1):
                self.root.after(0,lambda i=i,sym=sym:self.deep_progress.config(text=f"{i}/{len(syms)} | {sym}"))
                try: results.append(deep_analyze_symbol(sym))
                except Exception: pass
            results.sort(key=lambda x:x.get("score",0),reverse=True)
            def done():
                self.deep_table.delete(*self.deep_table.get_children())
                for r in results:self._deep_insert_result(r)
                if results:self.deep_report.insert("1.0",self._deep_report_text(results[0]))
                self.deep_progress.config(text=f"پایان | {len(results)}")
            self.root.after(0,done)
        threading.Thread(target=worker,daemon=True).start()



    def _fallback_ai_row(self, sym):
        try:
            sym=canonical(sym); price=safe_float(valid_symbols_map.get(sym)); df=get_candles_cached(sym,15,n=180,valid_symbols_map=valid_symbols_map)
            if df.empty or len(df)<40: return {"symbol":sym,"price":price,"decision":"WAIT","signal_state":"داده کندل ناکافی","score":50.0,"confidence":50.0,"history_ok":False,"data_status":"کندل ناکافی"}
            rsi=float(calculate_rsi(df)); macd=float(calculate_macd(df)); adx=float(calculate_adx(df)); dem=float(calculate_demarker(df)); vwap=calculate_vwap_daily(df)
            c=df.close; e9=float(c.ewm(span=9,adjust=False).mean().iloc[-1]); e21=float(c.ewm(span=21,adjust=False).mean().iloc[-1]); e50=float(c.ewm(span=50,adjust=False).mean().iloc[-1]); vr=float(df.volume.tail(20).mean()/max(df.volume.tail(60).mean(),1e-12)); score=50+(10 if c.iloc[-1]>e21 else -10)+(8 if e21>e50 else -8)+(8 if rsi<35 else -8 if rsi>65 else 0)+(6 if macd>0 else -6); score=float(np.clip(score,0,100)); decision="BUY" if score>=65 else "SELL" if score<=35 else "WAIT"
            return {"symbol":sym,"price":price,"decision":decision,"signal_state":decision,"score":score,"confidence":abs(score-50)+50,"regime":"bull" if e21>e50 else "bear","rsi":rsi,"macd":macd,"adx":adx,"stoch_k":50,"stoch_d":50,"cci":0,"mfi":50,"willr":-50,"bb_squeeze":False,"vwap":vwap,"ema9":e9,"ema21":e21,"ema50":e50,"volume_ratio":vr,"mtf":50,"method1":_deep_method_regime(df),"method2":_deep_method_liquidity(df),"method3":_deep_method_curvature(df),"nearest_support":float(df.low.tail(30).min()),"nearest_resistance":float(df.high.tail(30).max()),"entry":price,"stop_loss":None,"tp1":None,"tp2":None,"risk_reward":0.0,"history_ok":True,"data_status":"تحلیل تکنیکال پایه"}
        except Exception as e:
            return {"symbol":canonical(sym),"price":safe_float(valid_symbols_map.get(canonical(sym))),"decision":"WAIT","signal_state":"خطای داده","score":50.0,"confidence":50.0,"history_ok":False,"data_status":str(e)}

    def scan_smart_market(self):
        def worker():
            try:
                self.set_tab_activity("🧠 اجرای هوشمند", "در حال اسکن", "active")
                universe=get_smart_trade_universe(MAX_SMART_SCAN_SYMBOLS)
                rows=[]
                for sym in universe:
                    try:
                        r=unified_ai_decision(sym)
                        if not r or not r.get("price") or (not r.get("history_ok") and not r.get("rsi")):
                            r=self._fallback_ai_row(sym)
                        if r:
                            rows.append(r)
                            self._ensure_signal_from_decision(r, source="اجرای هوشمند")
                    except Exception:
                        pass
                rows.sort(key=lambda x:abs(float(x.get("score",50))-50)+float(x.get("confidence",0))*.15,reverse=True)
                rows=rows[:100]
                if self._auto_trade_state:
                    for r in rows:
                        d=r.get("decision","WAIT")
                        if d in ("BUY++","SELL++") and float(r.get("confidence",0))>=72 and float(r.get("risk_reward",0))>=1.8:
                            place_order(r["symbol"],"long" if d.startswith("BUY") else "short",r.get("trade_plan"),r.get("score",0),r.get("confidence",0))
                self.root.after(0,lambda rs=rows:self._show_smart_rows(rs))
            finally:
                self.root.after(0,lambda:self.smart_status.config(text="اسکن تکمیل شد"))
        self.smart_status.config(text="در حال تحلیل...")
        threading.Thread(target=worker,daemon=True).start()

    def _show_smart_rows(self, rows):
        for iid in self.smart_table.get_children(): self.smart_table.delete(iid)
        for r in rows:
            d=r.get("decision","WAIT"); tag=d.lower().replace("++","pp") if d!="WAIT" else "wait"
            vals=(
                r.get("symbol",""),r.get("signal_state",d),f"{r.get('score',0):.1f}",f"{r.get('confidence',0):.1f}%",
                format_price(r.get("entry")) if r.get("entry") else "—",
                format_price(r.get("stop_loss")) if r.get("stop_loss") else "—",
                format_price(r.get("tp1")) if r.get("tp1") else "—",
                format_price(r.get("tp2")) if r.get("tp2") else "—",
                f"{r.get('risk_reward',0):.2f}",
                "صعودی" if r.get("choch_bullish") else "نزولی" if r.get("choch_bearish") else "—",
                " | ".join(format_price(x) for x in r.get("eqh",[])) or "—",
                " | ".join(format_price(x) for x in r.get("eql",[])) or "—",
                f"B:{len(r.get('bullish_fvg',[]))}/S:{len(r.get('bearish_fvg',[]))}",
                f"{r.get('imbalance_score',0):+.1f}",
                "، ".join(r.get("candles",[])[:4]) or "—",
                r.get("nr_state",{}).get("nr","—"),
                r.get("order_block","—"),
                r.get("order_block","—")
            )
            self.smart_table.insert("","end",values=vals,tags=(tag,))
        if rows:
            b=rows[0]
            self.smart_report.delete("1.0","end")
            self.smart_report.insert("end",f"بهترین فرصت: {b.get('symbol')} | {b.get('decision')} | امتیاز {b.get('score',0):.1f} | اعتماد {b.get('confidence',0):.1f}%\n")
            self.smart_report.insert("end",f"Entry={format_price(b.get('entry'))} | SL={format_price(b.get('stop_loss'))} | TP1={format_price(b.get('tp1'))} | TP2={format_price(b.get('tp2'))} | RR={b.get('risk_reward',0):.2f}\n")
            self.smart_report.insert("end",f"CHoCH={'صعودی' if b.get('choch_bullish') else 'نزولی' if b.get('choch_bearish') else 'خنثی'} | EQH={b.get('eqh',[])} | EQL={b.get('eql',[])} | عدم‌تعادل={b.get('imbalance_score',0):+.1f}\n")
            self.smart_report.insert("end","کندل‌ها: "+("، ".join(b.get("candles",[])) or "—")+"\n")
            self.smart_report.insert("end",f"اندیکاتورها: Stoch={b.get('stoch_k',50):.1f}/{b.get('stoch_d',50):.1f} | CCI={b.get('cci',0):.1f} | MFI={b.get('mfi',50):.1f} | Williams%R={b.get('willr',-50):.1f} | BB={'SQUEEZE' if b.get('bb_squeeze') else 'عادی'}\n")
            self.smart_report.insert("end",f"EMA9/21/50={b.get('ema9',0):.4g}/{b.get('ema21',0):.4g}/{b.get('ema50',0):.4g} | VWAP={format_price(b.get('vwap')) if b.get('vwap') else '—'} | امتیاز همگرایی اندیکاتورها={b.get('indicator_confluence',0):+.1f}\n")
            self.smart_report.insert("end","منطق Narrow Range: NR4/NR7 باید با شکست High/Low، حجم، روند و ترجیحاً تایم‌فریم بالاتر تأیید شود؛ SL زیر/بالای کندل فشرده و TP با ATR/سطح یا حداقل R/R مناسب تنظیم می‌شود.\n")
            self.smart_report.insert("end","توجه: SL/TP ریسک را کاهش می‌دهد اما سود، وین‌ریت یا صفر شدن ریسک را تضمین نمی‌کند.")

    def show_researched_sources(self):
        try:
            self.smart_report.delete("1.0","end")
            self.smart_report.insert("end","منابع بررسی‌شده برای منطق موتور:\n")
            self.smart_report.insert("end","1) خانه سرمایه — Narrow Range: NR4/NR7، تأیید با حجم، شکست High/Low، SL نزدیک Low/High و TP با ATR/سطوح/RR.\n")
            self.smart_report.insert("end","2) خانه سرمایه — TradingView Signals: RSI، MACD، Bollinger، Fibonacci، MA/EMA، Stochastic، Ichimoku، CCI و ترکیب چند اندیکاتور.\n")
            self.smart_report.insert("end","3) مقاله سوم ارسالی: در این محیط متن کامل صفحه در دسترس نبود؛ بنابراین ادعای استخراج جزئیات آن وارد منطق قطعی نشده است. برای جلوگیری از نسبت‌دادن نادرست، فقط مواردی پیاده شده‌اند که از منابع قابل‌مشاهده پشتیبانی می‌شوند.\n")
            self.smart_report.insert("end","\nهدف: ساخت یک امتیاز همگرایی چندلایه، نه ادعای پیش‌بینی قطعی بازار.")
        except Exception: pass

    def save_selected_signal_snapshot(self):
        try:
            sel=self.smart_table.selection()
            if not sel: return
            sym=self.smart_table.item(sel[0],"values")[0]
            r=unified_ai_decision(sym)
            path=screenshot_current_signal(sym,r.get("decision","WAIT"),r.get("entry"),r.get("stop_loss"),r.get("tp1"),r.get("tp2"),r.get("score",0),r.get("confidence",0))
            if path: self.smart_status.config(text=f"ذخیره شد: {path}")
        except Exception as e:
            logging.debug("signal snapshot: %s",e)

    def _seed_ai_market_rows(self, symbols):
        """Populate every exchange symbol immediately; analysis updates the same row later."""
        try:
            for sym in symbols:
                p = safe_metric_value(valid_symbols_map.get(sym), 0.0)
                r = {
                    "symbol": sym,
                    "signal_state": "در انتظار تحلیل",
                    "decision": "WAIT",
                    "score": 0.0,
                    "confidence": 0.0,
                    "regime": "—",
                    "price": p,
                    "data_status": "قیمت بازار آماده؛ در انتظار کندل",
                    "history_ok": False
                }
                self._show_ai_row(r)
        except Exception as exc:
            logging.debug("_seed_ai_market_rows: %s", exc)

    def scan_ai_trader(self):
        all_symbols = list(SYMBOLS)
        if not all_symbols:
            self.ai_status.config(text="نمادی برای اسکن وجود ندارد")
            return

        # Show the full exchange universe immediately; analyze only a bounded subset.
        self._seed_ai_market_rows(all_symbols)
        symbols = get_smart_trade_universe(MAX_SMART_SCAN_SYMBOLS)

        try:
            self.ai_report.delete("1.0", "end")
        except Exception:
            pass

        def worker():
            self.set_tab_activity("مرکز معاملات هوشمند", f"شروع اسکن AI برای {len(symbols)} نماد", "active")
            results = []
            total = len(symbols)
            ok_history = 0
            no_history = 0
            for idx, sym in enumerate(symbols, 1):
                try:
                    r = unified_ai_decision(sym)
                    if not r or (not r.get("history_ok") and not r.get("rsi")):
                        r = self._fallback_ai_row(sym)
                    results.append(r)
                    self._ensure_signal_from_decision(r, source="AI Trader")
                    if r.get("history_ok"): ok_history += 1
                    else: no_history += 1
                    # UI is refreshed in small batches to keep tab switching responsive.
                    if idx % 10 == 0 or idx == total:
                        chunk = results[-10:]
                        self.root.after(0, lambda rows=list(chunk): [self._show_ai_row(x) for x in rows])
                except Exception as e:
                    logging.debug("AI scan failed %s: %s", sym, e)

                if idx % 10 == 0 or idx == total:
                    self.root.after(
                        0,
                        lambda i=idx, n=total:
                        self.ai_status.config(text=f"{i}/{n}")
                    )

            tradable = [
                r for r in results
                if r.get("decision") in {"BUY","BUY++","SELL","SELL++"}
            ]
            tradable.sort(key=lambda x: x.get("score", 50), reverse=True)

            report = [
                "=" * 92,
                "گزارش موتور یکپارچه AI Trader",
                "=" * 92,
                f"تعداد نمادهای اسکن‌شده: {len(results)}",
                f"فرصت‌های قابل معامله: {len(tradable)}",
                f"History موفق: {ok_history} | بدون History: {no_history}",
                "",
                "━━━ بهترین فرصت‌ها ━━━"
            ]
            if tradable:
                for r in tradable[:15]:
                    report.append(
                        f"{r['signal_state']} | {r['symbol']} | "
                        f"امتیاز {r['score']:.1f} | اعتماد {r['confidence']:.1f}% | "
                        f"رژیم {r['regime']} | R/R {r['risk_reward']:.2f}"
                    )
                    report.append(
                        f"  حمایت {format_price(r.get('nearest_support')) if r.get('nearest_support') else '—'} | "
                        f"مقاومت {format_price(r.get('nearest_resistance')) if r.get('nearest_resistance') else '—'} | "
                        f"حجم {r.get('volume_ratio',0):.2f}x | "
                        f"داده: {r.get('data_status','—')}"
                    )
                    if r.get("discovery_ready"):
                        report.append(
                            f"  کشف سیگنال: {r.get('discovery_direction','WAIT')} | "
                            f"لانگ {r.get('prob_long',0):.1f}% | "
                            f"شورت {r.get('prob_short',0):.1f}% | "
                            f"کیفیت {r.get('signal_quality',0):.1f}% | "
                            f"1h {r.get('trend_1h','—')}% | 4h {r.get('trend_4h','—')}% | 12h {r.get('trend_12h','—')}%"
                        )
            else:
                report.append(
                    "در اسکن فعلی فرصت معتبر بالاتر از آستانه پیدا نشد؛ WAIT به معنی نبود تأیید کافی است."
                )

            report += [
                "",
                "━━━ منطق سیگنال ━━━",
                "BUY / SELL = تایید اولیه چندلایه",
                "BUY++ / SELL++ = تایید اولیه + تایید ثانویه مستقل",
                "روش‌ها: رژیم بازار، فشار نقدینگی، خمیدگی حرکت",
                "فیلترها: قیمت، حجم، کندل، اندیکاتورها، حمایت/مقاومت، Order Block و MTF",
                "",
                "هشدار: امتیاز و سیگنال، احتمال و کیفیت فرصت را رتبه‌بندی می‌کنند و تضمین سود نیستند."
            ]

            self.root.after(
                0,
                lambda txt="\n".join(report): (
                    self._sort_ai_table_by_score(),
                    self.ai_report.delete("1.0","end"),
                    self.ai_report.insert("1.0",txt),
                    self.ai_report.see("1.0"),
                    self.ai_status.config(text="پایان اسکن"),
                    self.set_tab_activity("مرکز معاملات هوشمند", f"پایان اسکن AI | {len(results)} نماد", "done"),
                    self._refresh_trading_center_summary()
                )
            )

        threading.Thread(target=worker, daemon=True).start()


    def _show_ai_row(self, r):
        def display_num(v, digits=2):
            return fmt_metric(v, digits)

        vals = (
            r.get("symbol",""),
            r.get("signal_state", r.get("decision","WAIT")),
            display_num(r.get("score"), 1),
            f"{safe_metric_value(r.get('confidence'), 0):.1f}%",
            r.get("regime","—"),
            display_num(r.get("rsi"), 1),
            display_num(r.get("macd"), 5),
            display_num(r.get("adx"), 1),
            f"{display_num(r.get('stoch_k'),1)}/{display_num(r.get('stoch_d'),1)}",
            display_num(r.get("cci"), 1),
            display_num(r.get("mfi"), 1),
            display_num(r.get("willr"), 1),
            "SQUEEZE" if r.get("bb_squeeze") else "—",
            format_price(r.get("vwap")) if r.get("vwap") else "—",
            f"{fmt_metric(r.get('ema9'),4)}/{fmt_metric(r.get('ema21'),4)}/{fmt_metric(r.get('ema50'),4)}",
            f"{safe_metric_value(r.get('volume_ratio'),0):.2f}x" if r.get("volume_ratio") is not None else "—",
            display_num(r.get("mtf"), 1),
            display_num(r.get("method1"), 1),
            display_num(r.get("method2"), 1),
            display_num(r.get("method3"), 1),
            format_price(r.get("nearest_support")) if r.get("nearest_support") else "—",
            format_price(r.get("nearest_resistance")) if r.get("nearest_resistance") else "—",
            format_price(r.get("entry")) if r.get("entry") else "—",
            format_price(r.get("stop_loss")) if r.get("stop_loss") else "—",
            format_price(r.get("tp1")) if r.get("tp1") else "—",
            format_price(r.get("tp2")) if r.get("tp2") else "—",
            f"{safe_metric_value(r.get('risk_reward'),0):.2f}",
        )
        decision = r.get("decision","WAIT")
        tag = (
            "buypp" if decision == "BUY++" else
            "buy" if decision == "BUY" else
            "sellpp" if decision == "SELL++" else
            "sell" if decision == "SELL" else
            "wait"
        )
        try:
            iid = None
            symbol = r.get("symbol","")
            for existing in self.ai_table.get_children():
                ev = self.ai_table.item(existing, "values")
                if ev and str(ev[0]) == str(symbol):
                    iid = existing
                    break
            if iid:
                self.ai_table.item(iid, values=vals, tags=(tag,))
            else:
                iid = self.ai_table.insert("", "end", values=vals, tags=(tag,))

            if not r.get("history_ok", False):
                # Keep status truthful, but do not erase useful scan progress.
                self.ai_status.config(text=f"{symbol} | {r.get('data_status','داده ناقص')}")
            score = safe_metric_value(r.get("score"), 0)
            if score >= 85:
                self.ai_table.selection_set(iid)
            self._refresh_trading_center_summary()
        except Exception as exc:
            logging.debug("_show_ai_row failed: %s", exc)
            return

    def _schedule_ai_scan(self):
        try:
            if getattr(self, "running", False):
                self.scan_ai_trader()
        except Exception:
            pass
        finally:
            try:
                self.root.after(65000, self._schedule_ai_scan)
            except Exception:
                pass

    def _sort_ai_table_by_score(self):
        try:
            items=[]
            for iid in self.ai_table.get_children():
                vals=self.ai_table.item(iid,"values")
                try: score=float(vals[2])
                except Exception: score=-1e9
                items.append((score,iid))
            items.sort(key=lambda x:x[0], reverse=True)
            for pos,(_,iid) in enumerate(items):
                self.ai_table.move(iid,"",pos)
        except Exception:
            pass

    def refresh_ml_tab(self):
        try:
            self.ml_table.delete(*self.ml_table.get_children())
            active = [s for s in SYMBOLS if s not in pruned_symbols][:20]
            for sym in active:
                try:
                    pred = predict_with_ensemble(sym)
                    prob = pred.get("prob_up", 0.0) * 100
                    move = pred.get("expected_move", 0.0)
                    conf = pred.get("model_confidence", 0.0) * 100
                    mtf = calculate_mtf_alignment(sym) * 100
                    df = get_candles_cached(sym, 60, n=50, valid_symbols_map=valid_symbols_map)
                    rsi = calculate_rsi(df) if not df.empty else "—"
                    macd = calculate_macd(df) if not df.empty else "—"
                    adx = calculate_adx(df) if not df.empty else "—"
                    if prob >= 60 and move >= 1.0:
                        rec = "خرید قوی" if prob >= 70 else "خرید"
                    elif prob <= 40 and move >= 1.0:
                        rec = "فروش قوی" if prob <= 30 else "فروش"
                    else:
                        rec = "نگهداری"
                    vals = (sym, f"{prob:.1f}%", f"{move:.2f}", f"{conf:.1f}%", f"{mtf:.0f}%", rsi, macd, adx, rec)
                    iid = self.ml_table.insert("", "end", values=vals)
                    if "خرید" in rec:
                        self.ml_table.item(iid, tags=("long",))
                    elif "فروش" in rec:
                        self.ml_table.item(iid, tags=("short",))
                except Exception:
                    continue
        except Exception:
            pass

    def refresh_structure_tab(self):
        try:
            selected = self.main_table.selection()
            combo_sym = ""
            try:
                combo_sym = self.structure_symbol_combo.get().strip()
            except Exception:
                pass

            sym = ""
            if selected:
                vals = self.main_table.item(selected[0]).get("values", [])
                if vals:
                    sym = str(vals[0]).strip()
            if not sym:
                sym = combo_sym or (SYMBOLS[0] if SYMBOLS else "")

            if not sym:
                self.structure_text.delete("1.0", "end")
                self.structure_text.insert("1.0", "هیچ نمادی در دسترس نیست.")
                return

            self.structure_symbol_combo.set(sym)

            df = get_candles_cached(
                sym, 60, n=300, valid_symbols_map=valid_symbols_map
            )
            if df.empty or len(df) < 50:
                self.structure_text.delete("1.0", "end")
                self.structure_text.insert(
                    "1.0",
                    f"برای {sym} کندل کافی برای تحلیل ساختاری دریافت نشد."
                )
                return

            price = safe_float(df["close"].iloc[-1])
            prev_price = safe_float(df["close"].iloc[-2])
            change = (
                (price - prev_price) / prev_price * 100
                if prev_price > 0 else 0.0
            )

            rsi = calculate_rsi(df)
            macd = calculate_macd(df)
            adx = calculate_adx(df)
            dem = calculate_demarker(df)
            ub, mid, lb = calculate_bollinger_bands(df)
            vwap = calculate_vwap_daily(df)
            if vwap is None:
                try:
                    tp = (df["high"] + df["low"] + df["close"]) / 3.0
                    den = float(df["volume"].sum())
                    vwap = float((tp * df["volume"]).sum() / den) if den > 0 else None
                except Exception:
                    vwap = None

            sh, sl = detect_swing_points(df)
            bos = detect_bos_choch(df, sh, sl)
            bullish_ob, bearish_ob = detect_order_blocks(df, sh, sl)
            fvg_bull, fvg_bear = detect_fvg(df)
            sweep_high, sweep_low = detect_liquidity_sweeps(df, sh, sl)

            trend = "خنثی"
            if len(df) >= 50:
                ema20 = df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
                ema50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
                if price > ema20 > ema50:
                    trend = "صعودی قوی"
                elif price > ema20:
                    trend = "صعودی"
                elif price < ema20 < ema50:
                    trend = "نزولی قوی"
                elif price < ema20:
                    trend = "نزولی"

            structure_bias = "خنثی"
            if bos.get("bos_bullish") or bos.get("choch_bullish"):
                structure_bias = "غلبه خریداران"
            elif bos.get("bos_bearish") or bos.get("choch_bearish"):
                structure_bias = "غلبه فروشندگان"

            ob_imbalance = 0.0
            try:
                ob_imbalance = calculate_orderbook_imbalance(sym)
            except Exception:
                pass

            mtf = 0.5
            try:
                mtf = calculate_mtf_alignment(sym)
            except Exception:
                pass

            last = df.iloc[-1]
            candle_direction = "صعودی" if last["close"] >= last["open"] else "نزولی"
            volume_ratio = get_volume_ratio_from_candles(
                df, recent_bars=6, prev_bars=48
            )

            support = min(
                [float(df["low"].iloc[i]) for i in sl[-5:]]
                or [float(df["low"].tail(50).min())]
            )
            resistance = max(
                [float(df["high"].iloc[i]) for i in sh[-5:]]
                or [float(df["high"].tail(50).max())]
            )

            lines = [
                "=" * 78,
                f"تحلیل جامع ساختار بازار — {sym}",
                "=" * 78,
                f"زمان تحلیل: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"بازه تحلیل: 60 دقیقه | تعداد کندل: {len(df)}",
                "",
                "━━━ وضعیت فعلی بازار ━━━",
                f"قیمت فعلی: {format_price(price)}",
                f"تغییر آخرین کندل: {change:+.2f}٪",
                f"جهت آخرین کندل: {candle_direction}",
                f"روند: {trend}",
                f"سوگیری ساختاری: {structure_bias}",
                f"نسبت حجم اخیر: {volume_ratio:.2f}x",
                f"فشار اردربوک: {ob_imbalance:+.3f}  (مثبت = غلبه تقاضا)",
                f"هم‌راستایی چندبازه‌ای: {mtf:.0%}",
                "",
                "━━━ شاخص‌های تکنیکال ━━━",
                f"RSI(14): {rsi:.1f}",
                f"MACD: {macd:.6f}",
                f"DeMarker: {dem:.3f}",
                f"ADX(14): {adx:.1f}",
                f"VWAP: {format_price(vwap) if vwap else '—'}",
                f"باند بالایی بولینگر: {format_price(ub) if ub else '—'}",
                f"باند میانی بولینگر: {format_price(mid) if mid else '—'}",
                f"باند پایینی بولینگر: {format_price(lb) if lb else '—'}",
                "",
                "━━━ ساختار بازار ━━━",
                f"تعداد Swing High: {len(sh)}",
                f"تعداد Swing Low: {len(sl)}",
                f"BOS صعودی: {'بله' if bos.get('bos_bullish') else 'خیر'}",
                f"BOS نزولی: {'بله' if bos.get('bos_bearish') else 'خیر'}",
                f"CHoCH صعودی: {'بله' if bos.get('choch_bullish') else 'خیر'}",
                f"CHoCH نزولی: {'بله' if bos.get('choch_bearish') else 'خیر'}",
                f"آخرین مقاومت ساختاری: {format_price(bos.get('last_swing_high')) if bos.get('last_swing_high') else format_price(resistance)}",
                f"آخرین حمایت ساختاری: {format_price(bos.get('last_swing_low')) if bos.get('last_swing_low') else format_price(support)}",
                "",
                "━━━ نواحی سفارش / نقدینگی ━━━",
                f"Order Block صعودی: {len(bullish_ob)}",
                f"Order Block نزولی: {len(bearish_ob)}",
                f"FVG صعودی: {len(fvg_bull)}",
                f"FVG نزولی: {len(fvg_bear)}",
                f"شکار نقدینگی سقف: {'بله' if sweep_high else 'خیر'}",
                f"شکار نقدینگی کف: {'بله' if sweep_low else 'خیر'}",
            ]

            if bullish_ob:
                lines.append("")
                lines.append("آخرین Order Blockهای صعودی:")
                for ob in bullish_ob[-3:]:
                    lines.append(
                        f"  محدوده: {format_price(ob.get('low'))} تا {format_price(ob.get('high'))}"
                    )

            if bearish_ob:
                lines.append("")
                lines.append("آخرین Order Blockهای نزولی:")
                for ob in bearish_ob[-3:]:
                    lines.append(
                        f"  محدوده: {format_price(ob.get('low'))} تا {format_price(ob.get('high'))}"
                    )

            if fvg_bull:
                lines.append("")
                lines.append("آخرین FVGهای صعودی:")
                for gap in fvg_bull[-3:]:
                    lines.append(
                        f"  محدوده: {format_price(gap.get('bottom'))} تا {format_price(gap.get('top'))} "
                        f"| اندازه: {gap.get('gap_pct', 0):.3f}٪"
                    )

            if fvg_bear:
                lines.append("")
                lines.append("آخرین FVGهای نزولی:")
                for gap in fvg_bear[-3:]:
                    lines.append(
                        f"  محدوده: {format_price(gap.get('bottom'))} تا {format_price(gap.get('top'))} "
                        f"| اندازه: {gap.get('gap_pct', 0):.3f}٪"
                    )

            # A simple, explicit conclusion in Persian.
            lines += [
                "",
                "━━━ جمع‌بندی تحلیلی ━━━",
            ]
            if bos.get("bos_bullish") or (rsi < 40 and macd > 0):
                lines.append("سناریوی صعودی: نشانه‌هایی از تقویت تقاضا وجود دارد.")
            elif bos.get("bos_bearish") or (rsi > 60 and macd < 0):
                lines.append("سناریوی نزولی: نشانه‌هایی از تقویت عرضه وجود دارد.")
            else:
                lines.append("سناریوی اصلی: بازار در وضعیت خنثی/انتظار قرار دارد.")

            if adx >= 25:
                lines.append("قدرت روند: مناسب؛ ADX نشان‌دهنده روند قابل‌توجه است.")
            else:
                lines.append("قدرت روند: ضعیف تا متوسط؛ احتمال نوسان و رفت‌وبرگشت بیشتر است.")

            if volume_ratio >= 1.5:
                lines.append("تأیید حجم: افزایش حجم مشاهده می‌شود.")
            else:
                lines.append("تأیید حجم: افزایش حجم قدرتمند مشاهده نشده است.")

            lines.append("")
            lines.append(
                "هشدار: این گزارش تحلیلی است و به‌تنهایی مبنای معامله قطعی نیست."
            )

            self.structure_text.delete("1.0", "end")
            self.structure_text.insert("1.0", "\n".join(lines))
            self.structure_text.see("1.0")

        except Exception as e:
            logging.exception("refresh_structure_tab error: %s", e)
            self.structure_text.delete("1.0", "end")
            self.structure_text.insert(
                "1.0",
                f"خطا در تحلیل ساختار بازار:\n{e}"
            )


    def update_risk_tab(self):
        try:
            self.risk_table.delete(*self.risk_table.get_children())
            with state_lock:
                trades = list(auto_trade_log)

            if trades:
                mc = monte_carlo_simulation(trades)
                metrics = [
                    ("سرمایه فعلی", f"{_equity_curve[-1]:.2f} USDT", "正常"),
                    ("حداکثر افت", f"{mc['worst_max_dd']:.2f}%", "WARNING" if mc['worst_max_dd'] > MAX_DRAWDOWN_PCT else "OK"),
                    ("احتمال ورشکستگی", f"{mc['prob_ruin']:.2%}", "WARNING" if mc['prob_ruin'] > 0.05 else "OK"),
                    ("نرخ برد", f"{mc['sharpe']:.2f}", "OK" if mc['sharpe'] > 1 else "WARNING"),
                    ("بازده میانه", f"{mc['median_final']:.2f} USDT", "OK" if mc['median_final'] > 1000 else "WARNING"),
                ]
            else:
                metrics = [
                    ("سرمایه فعلی", "1000.00 USDT", "OK"),
                    ("حداکثر افت", "0.00%", "OK"),
                    ("احتمال ورشکستگی", "0.00%", "OK"),
                    ("نرخ برد", "—", "—"),
                    ("بازده میانه", "1000.00 USDT", "OK"),
                ]

            for m, v, s in metrics:
                iid = self.risk_table.insert("", "end", values=(m, v, s))
                if s == "WARNING":
                    self.risk_table.item(iid, tags=("warning",))
                elif s == "OK":
                    self.risk_table.item(iid, tags=("ok",))
            self.risk_table.tag_configure("warning", background="#991b1b", foreground="white")
            self.risk_table.tag_configure("ok", background="#166534", foreground="white")
        except Exception:
            pass

    def run_monte_carlo(self):
        try:
            with state_lock:
                trades = list(auto_trade_log)
            if not trades:
                messagebox.showinfo("مونت‌کارلو", "هنوز معامله‌ای انجام نشده است")
                return
            mc = monte_carlo_simulation(trades)
            msg = (
                f"نتایج شبیه‌سازی مونت‌کارلو ({MONTE_CARLO_SIMS} بار):\n\n"
                f"سرمایه میانه: {mc['median_final']:.2f} USDT\n"
                f"بدترین سرمایه: {mc['worst_final']:.2f} USDT\n"
                f"بهترین سرمایه: {mc['best_final']:.2f} USDT\n"
                f"حداکثر افت میانه: {mc['median_max_dd']:.2f}%\n"
                f"بدترین افت: {mc['worst_max_dd']:.2f}%\n"
                f"احتمال ورشکستگی: {mc['prob_ruin']:.2%}\n"
                f"شارپ: {mc['sharpe']:.3f}"
            )
            messagebox.showinfo("مونت‌کارلو", msg)
        except Exception as e:
            messagebox.showerror("خطا", str(e))


    def _autotrade_action_click(self, event):
        try:
            row = self.autotrade_table.identify_row(event.y)
            col = self.autotrade_table.identify_column(event.x)
            if not row or col != f"#{len(self.autotrade_table['columns'])}": return
            vals = self.autotrade_table.item(row, "values")
            if not vals or not vals[0] or vals[0] == "—": return
            symbol = canonical(vals[1])
            ts = vals[0]
            with state_lock:
                candidates = [o for o in auto_trade_log if canonical(o.get("symbol")) == symbol and str(o.get("time")) == str(ts)]
                order_ref = candidates[0] if candidates else None
            if order_ref is None:
                messagebox.showinfo("حذف معامله", "رکورد معامله پیدا نشد."); return
            if order_ref.get("result") is None or not order_ref.get("exit_time"):
                messagebox.showwarning("حذف معامله", "این معامله هنوز باز است. ابتدا با «فروش فوری / بستن» آن را ببندید؛ حذف معامله باز عمداً مسدود است تا موقعیت واقعی بدون مدیریت نماند.")
                return
            if not messagebox.askyesno("حذف معامله", f"رکورد معامله {symbol} در {ts} حذف شود؟"):
                return
            key = _trade_history_key(order_ref)
            with state_lock:
                auto_trade_log[:] = [o for o in auto_trade_log if _trade_history_key(o) != key]
                trade_history[:] = [o for o in trade_history if _trade_history_key(o) != key]
                rows = list(trade_history)
            self._rewrite_trade_history_file(rows)
            self.refresh_tables(); self.refresh_stats_dashboard()
            logging.warning("[TRADE RECORD DELETED] %s", key)
        except Exception as exc:
            logging.exception("autotrade delete click failed: %s", exc)
            messagebox.showerror("حذف معامله", str(exc))

    def _signal_action_click(self, event):
        """Handle the per-signal red close button rendered in the last Treeview column."""
        try:
            row = self.signals_table.identify_row(event.y)
            col = self.signals_table.identify_column(event.x)
            if not row or col != f"#{len(self.signals_table['columns'])}":
                return
            vals = self.signals_table.item(row, "values")
            if not vals or not vals[0] or vals[0] == "—":
                return
            signal_id = str(vals[0])
            symbol = canonical(vals[1]) if len(vals) > 1 else ""
            side_txt = str(vals[9]).strip() if len(vals) > 9 else ""
            side = "long" if side_txt in {"لانگ", "long", "BUY", "خرید"} else "short"
            self.force_close_signal_trade(signal_id, symbol, side)
        except Exception as exc:
            logging.exception("signal action click failed: %s", exc)

    def force_close_signal_trade(self, signal_id, symbol, side):
        """Immediately close the newest open trade matching a signal's symbol/side."""
        try:
            with state_lock:
                candidates = [o for o in auto_trade_log
                              if o.get("result") is None
                              and canonical(o.get("symbol")) == canonical(symbol)
                              and str(o.get("side", "")).lower() == str(side).lower()]
                candidates.sort(key=lambda x: safe_float(x.get("ts")), reverse=True)
                order_ref = candidates[0] if candidates else None
            if order_ref is None:
                messagebox.showinfo("فروش فوری / بستن", f"برای {symbol} معامله باز پیدا نشد.")
                return
            if _real_trading_is_enabled():
                if not _real_submit_exit(order_ref, "MANUAL_FORCE_CLOSE"):
                    messagebox.showerror("بستن معامله", f"بستن واقعی {symbol} ناموفق بود. جزئیات در لاگ ثبت شده است.")
                    return
            current = safe_float(order_ref.get("current_price")) or safe_float(order_ref.get("entry_price"))
            entry = safe_float(order_ref.get("entry_price"))
            lev = max(1.0, safe_float(order_ref.get("leverage")) or 1.0)
            raw = ((current-entry)/entry*100.0) if side == "long" and entry > 0 else ((entry-current)/entry*100.0 if entry > 0 else 0.0)
            with state_lock:
                order_ref["exit_price"] = current
                order_ref["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                order_ref["final_pct"] = round(raw * lev, 4)
                order_ref["result"] = "فروش فوری"
                order_ref["status"] = "بسته (دستی)"
                order_ref["real_exit_reason"] = "MANUAL_FORCE_CLOSE"
                order_ref["pnl_usdt"] = _calculate_trade_pnl_usdt(order_ref)
            try:
                _v18_log_trade_closed(order_ref, "MANUAL_FORCE_CLOSE")
            except Exception:
                pass
            try:
                save_closed_trade(order_ref)
            except Exception:
                pass
            self.root.after(0, self.refresh_tables)
            self.root.after(0, self.refresh_stats_dashboard)
            logging.warning("[MANUAL FORCE CLOSE] signal=%s symbol=%s side=%s pnl=%s", signal_id, symbol, side, order_ref.get("pnl_usdt"))
        except Exception as exc:
            logging.exception("manual force close failed: %s", exc)
            messagebox.showerror("بستن معامله", str(exc))

    def _trade_datetime(self, t):
        try:
            if t.get("ts") not in (None, ""):
                return datetime.fromtimestamp(float(t.get("ts")))
            return datetime.strptime(str(t.get("exit_time"))[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    def _rewrite_trade_history_file(self, rows):
        import csv
        tmp = TRADE_HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRADE_HISTORY_FIELDS)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in TRADE_HISTORY_FIELDS})
        os.replace(tmp, TRADE_HISTORY_FILE)

    def _delete_trade_rows(self, predicate, label):
        try:
            with state_lock:
                current_closed = [dict(o) for o in auto_trade_log if o.get("final_pct") is not None and o.get("exit_time")]
                deleted_keys = set()
                kept_history = []
                deleted = 0
                for row in trade_history:
                    if predicate(row):
                        deleted_keys.add(_trade_history_key(row)); deleted += 1
                    else:
                        kept_history.append(row)
                # Remove matching closed in-memory records too, while NEVER deleting open trades.
                for o in list(auto_trade_log):
                    if o.get("final_pct") is not None and o.get("exit_time") and predicate(o):
                        k = _trade_history_key(o)
                        if k not in deleted_keys:
                            deleted += 1
                        deleted_keys.add(k)
                trade_history[:] = kept_history
                for i in range(len(auto_trade_log)-1, -1, -1):
                    o = auto_trade_log[i]
                    if o.get("final_pct") is not None and o.get("exit_time") and _trade_history_key(o) in deleted_keys:
                        auto_trade_log.pop(i)
                remaining = list(trade_history)
            self._rewrite_trade_history_file(remaining)
            self.refresh_stats_dashboard()
            self.refresh_tables()
            messagebox.showinfo("حذف معاملات", f"{deleted} معامله بسته‌شده حذف شد.\n{label}")
        except Exception as exc:
            logging.exception("trade history deletion failed: %s", exc)
            messagebox.showerror("حذف معاملات", str(exc))

    def delete_selected_trade_history(self):
        try:
            sel = self.stats_table.selection()
            if not sel:
                messagebox.showinfo("حذف معامله", "ابتدا یک ردیف در جدول آمار انتخاب کنید.")
                return
            # Individual trade deletion is done from the auto-trade table when a row is selected.
            # If stats table is selected, delete matching symbol only for the currently filtered result.
            vals = self.stats_table.item(sel[0], "values")
            symbol = canonical(vals[0]) if vals else ""
            if not symbol:
                return
            if not messagebox.askyesno("تأیید حذف", f"تمام معاملات بسته‌شده {symbol} با فیلتر فعلی حذف شوند؟"):
                return
            period, result = self.stats_period.get(), self.stats_result.get()
            self._delete_trade_rows(lambda t: canonical(t.get("symbol")) == symbol and
                                    (result == "همه" or str(t.get("result")) == result) and
                                    any(True for _ in [t] if self._trade_in_period(t, period)),
                                    f"نماد: {symbol} | بازه: {period}")
        except Exception as exc:
            messagebox.showerror("حذف معامله", str(exc))

    def _trade_in_period(self, t, period):
        dt = self._trade_datetime(t); now = datetime.now()
        if dt is None: return False
        if period == "روزانه": return dt >= now - timedelta(days=1)
        if period == "هفتگی": return dt >= now - timedelta(days=7)
        if period == "ماهانه": return dt >= now - timedelta(days=30)
        if period == "سالانه": return dt >= now - timedelta(days=365)
        return True

    def delete_trade_history_range(self):
        try:
            f = datetime.strptime(self.stats_from_var.get().strip(), "%Y-%m-%d %H:%M:%S")
            t = datetime.strptime(self.stats_to_var.get().strip(), "%Y-%m-%d %H:%M:%S")
            if t < f: raise ValueError("زمان پایان باید بعد از زمان شروع باشد")
            symbol, result = self.stats_symbol.get(), self.stats_result.get()
            if not messagebox.askyesno("تأیید صفر کردن", f"معاملات بسته‌شده از {f} تا {t} حذف شوند؟"):
                return
            self._delete_trade_rows(lambda x: (self._trade_datetime(x) is not None and f <= self._trade_datetime(x) <= t)
                                    and (symbol == "همه" or canonical(x.get("symbol")) == canonical(symbol))
                                    and (result == "همه" or str(x.get("result")) == result),
                                    f"بازه: {f} تا {t}")
        except Exception as exc:
            messagebox.showerror("بازه زمانی", "فرمت صحیح: YYYY-MM-DD HH:MM:SS\n" + str(exc))

    def delete_all_trade_history(self):
        try:
            if not messagebox.askyesno("هشدار", "همه معاملات بسته‌شده از تاریخچه و آمار حذف شوند؟\nمعاملات باز دست‌نخورده می‌مانند."):
                return
            self._delete_trade_rows(lambda t: True, "همه معاملات بسته‌شده")
        except Exception as exc:
            messagebox.showerror("حذف همه", str(exc))

    def refresh_stats_dashboard(self):
        try:
            period=self.stats_period.get(); symbol=self.stats_symbol.get(); result=self.stats_result.get()
            trades=list(filter_trade_history(period,symbol,result)); sm=summarize_trades(trades)
            vals=[f"{sm['trades']}",f"{sm['wins']}",f"{sm['losses']}",f"+{sum(max(0,safe_float(t.get('pnl_usdt'))) for t in trades):.2f} USDT",f"-{abs(sum(min(0,safe_float(t.get('pnl_usdt'))) for t in trades)):.2f} USDT",f"{sm['pnl']:+.2f} USDT",f"{sm['win_rate']:.1f}%",f"{sm['avg']:+.3f} USDT"]
            for w,v in zip(self.stats_cards,vals): w.config(text=v,fg="#22c55e" if "+" in v and "-" not in v else ("#ef4444" if v.startswith("-") else "white"))
            by={}
            for t in trades:
                k=canonical(t.get("symbol")); by.setdefault(k,[]).append(t)
            ranked=sorted((summarize_trades(v)|{"symbol":k} for k,v in by.items()), key=lambda x:x["pnl"], reverse=True)
            bestp=ranked[0]["symbol"] if ranked else "—"; worstp=min(ranked,key=lambda x:x["pnl"])["symbol"] if ranked else "—"; most=max(ranked,key=lambda x:x["trades"])["symbol"] if ranked else "—"
            self.stats_text.delete("1.0","end")
            self.stats_text.insert("1.0",f"بازه: {period} | معاملات: {sm['trades']} | سود: {sm['wins']} | ضرر: {sm['losses']}\nسود ناخالص: +{sum(max(0,safe_float(t.get('pnl_usdt'))) for t in trades):.2f} USDT | زیان ناخالص: -{abs(sum(min(0,safe_float(t.get('pnl_usdt'))) for t in trades)):.2f} USDT | خالص: {sm['pnl']:+.2f} USDT\nبیشترین سود: {bestp} | بیشترین ضرر: {worstp} | بیشترین تعداد معامله: {most}\nبهترین معامله: {sm['best']:+.3f} USDT | بدترین معامله: {sm['worst']:+.3f} USDT | میانگین: {sm['avg']:+.3f} USDT")
            self.stats_table.delete(*self.stats_table.get_children())
            for x in ranked:
                iid=self.stats_table.insert("","end",values=(x["symbol"],x["trades"],x["wins"],x["losses"],f"{x['pnl']:+.3f}",f"{x['avg']:+.3f}",f"{x['win_rate']:.1f}%",f"{x['best']:+.3f}",f"{x['worst']:+.3f}"),tags=("profit" if x["pnl"]>0 else "loss",))
        except Exception as e: logging.exception("stats dashboard failed: %s",e)

    def export_trade_history(self):
        try:
            import csv
            path=str(BASE_DIR/"trade_history_export.csv"); rows=get_completed_trades()
            if not rows: messagebox.showinfo("آمار معاملات","هنوز معامله بسته‌شده‌ای ثبت نشده است"); return
            fields=sorted(set().union(*(r.keys() for r in rows)))
            with open(path,"w",encoding="utf-8-sig",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
            messagebox.showinfo("Export CSV",f"فایل ساخته شد:\n{path}")
        except Exception as e: messagebox.showerror("Export CSV",str(e))

    def _load_backtest_history(self, symbol: str, bars: int = 2000, resolution: int = 60) -> pd.DataFrame:
        """Fetch long historical data in bounded chunks and merge them."""
        bars = max(500, min(10000, int(bars)))
        now = int(time.time())
        chunk_bars = 450
        all_frames = []
        remaining = bars
        cursor_end = now
        seen_first = None

        while remaining > 0:
            this_bars = min(chunk_bars, remaining)
            cursor_start = cursor_end - this_bars * resolution * 60
            j = robust_history_udf(
                symbol, resolution, cursor_start, cursor_end,
                valid_symbols=set(valid_symbols_map.keys()) if valid_symbols_map else None
            )
            if not j:
                break
            try:
                o = j.get("o") or []
                h = j.get("h") or []
                l = j.get("l") or []
                c = j.get("c") or []
                v = j.get("v") or []
                ts = j.get("t") or []
                n = min(len(o), len(h), len(l), len(c), len(v))
                if n < 10:
                    break
                data = {
                    "open": o[-n:], "high": h[-n:], "low": l[-n:],
                    "close": c[-n:], "volume": v[-n:]
                }
                frame = pd.DataFrame(data).astype(float)
                if ts:
                    frame.index = pd.to_datetime(ts[-n:], unit="s")
                else:
                    frame.index = pd.date_range(
                        end=pd.to_datetime(cursor_end, unit="s"),
                        periods=n, freq=f"{resolution}min"
                    )
                all_frames.append(frame)
                earliest = int(frame.index[0].timestamp())
                if seen_first is not None and earliest >= seen_first:
                    break
                seen_first = earliest
                got = len(frame)
                remaining -= got
                cursor_end = earliest - resolution * 60
                if got < max(20, int(this_bars * 0.7)):
                    break
            except Exception:
                break

        if not all_frames:
            return pd.DataFrame()
        df = pd.concat(all_frames).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        if len(df) > bars:
            df = df.tail(bars)
        return df

    def run_backtest_dialog(self):
        def worker(symbol, bars):
            try:
                self.root.after(
                    0, lambda: self.bt_progress.config(
                        text=f"در حال دریافت {symbol} ({bars} کندل)..."
                    )
                )
                bars = max(500, min(5000, int(bars)))
                df = self._load_backtest_history(symbol, bars=bars, resolution=60)
                if df.empty or len(df) < 300:
                    raise RuntimeError(
                        f"داده کافی برای {symbol} دریافت نشد. تعداد کندل: {len(df)}"
                    )

                signals = []
                step = 1

                # Historical version of the live score. We deliberately do not
                # call the live signal engine here; that would be too slow and
                # would mix current-state APIs into the historical test.
                for i in range(80, len(df) - 8, step):
                    w = df.iloc[:i+1]
                    rsi = calculate_rsi(w)
                    macd = calculate_macd(w)
                    dem = calculate_demarker(w)

                    score = 0.0
                    if rsi < 42 and macd > 0:
                        score += 8
                    elif rsi < 42:
                        score += 4
                    elif rsi > 58 and macd < 0:
                        score -= 8
                    elif rsi > 58:
                        score -= 4

                    if dem < 0.33:
                        score += 2
                    elif dem > 0.67:
                        score -= 2

                    if abs(score) < 5:
                        continue

                    # Add a volume confirmation without using future bars.
                    vr = get_volume_ratio_from_candles(
                        w, recent_bars=3, prev_bars=24
                    )
                    if vr >= 1.2:
                        score += 1 if score > 0 else -1

                    pred = max(
                        0.5,
                        float(estimate_predicted_move(score, 0.0, 1))
                    )
                    signals.append({
                        "time": w.index[-1],
                        "entry": float(w["close"].iloc[-1]),
                        "side": "لانگ" if score > 0 else "شورت",
                        "predicted_move_pct": pred,
                        "stop_loss_pct": max(
                            0.5, float(STOP_LOSS_PCT_DEFAULT)
                        )
                    })

                bt = VectorizedBacktester()
                result = bt.run(df, signals)

                # Extra diagnostics from completed trades.
                trade_returns = []
                wins = []
                losses = []
                try:
                    trade_returns = [float(x.get("return", 0.0)) for x in bt.trades]
                    wins = [x for x in trade_returns if x > 0]
                    losses = [x for x in trade_returns if x <= 0]
                except Exception:
                    pass

                gross_profit = sum(wins)
                gross_loss = abs(sum(losses))
                profit_factor = (
                    gross_profit / gross_loss
                    if gross_loss > 0 else
                    float("inf") if gross_profit > 0 else 0.0
                )
                avg_trade = (
                    sum(trade_returns) / len(trade_returns)
                    if trade_returns else 0.0
                )
                best_trade = max(trade_returns) if trade_returns else 0.0
                worst_trade = min(trade_returns) if trade_returns else 0.0

                # Buy & hold comparison.
                first_close = float(df["close"].iloc[0])
                last_close = float(df["close"].iloc[-1])
                buy_hold = (
                    (last_close - first_close) / first_close * 100.0
                    if first_close > 0 else 0.0
                )

                report = [
                    "=" * 82,
                    f"گزارش جامع بک‌تست — {symbol}",
                    "=" * 82,
                    f"تاریخ اجرا: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"بازه: {df.index[0]} تا {df.index[-1]}",
                    f"تعداد کندل استفاده‌شده: {len(df)}",
                    "",
                    "━━━ آمار معاملات ━━━",
                    f"تعداد سیگنال‌های تاریخی: {len(signals)}",
                    f"تعداد معاملات تکمیل‌شده: {result.get('trades', 0)}",
                    f"نرخ برد: {result.get('win_rate', 0.0):.2f}٪",
                    f"نرخ باخت: {100.0 - result.get('win_rate', 0.0):.2f}٪",
                    f"میانگین بازده هر معامله: {avg_trade:.3f}٪",
                    f"بهترین معامله: {best_trade:.3f}٪",
                    f"بدترین معامله: {worst_trade:.3f}٪",
                    "",
                    "━━━ عملکرد سرمایه ━━━",
                    f"بازده کل استراتژی: {result.get('total_return', 0.0):.2f}٪",
                    f"بازده خرید و نگهداری: {buy_hold:.2f}٪",
                    f"شارپ: {result.get('sharpe', 0.0):.3f}",
                    f"حداکثر افت سرمایه: {result.get('max_drawdown', 0.0):.2f}٪",
                    f"سود ناخالص معاملات: {gross_profit:.3f}٪",
                    f"زیان ناخالص معاملات: {gross_loss:.3f}٪",
                    f"Profit Factor: {profit_factor:.3f}" if profit_factor != float("inf")
                    else "Profit Factor: بی‌نهایت (بدون معامله زیان‌ده)",
                    "",
                    "━━━ پارامترهای تست ━━━",
                    f"سرمایه اولیه: {bt.initial_capital:.2f} USDT",
                    f"اهرم استفاده‌شده: {LEVERAGE}x",
                    f"حد ضرر پایه: {STOP_LOSS_PCT_DEFAULT:.2f}٪",
                    f"افق خروج هر معامله: حدود 24 کندل",
                    "",
                    "نتیجه:",
                    (
                        "✅ استراتژی در این بازه سودده بوده است."
                        if result.get("total_return", 0) > 0
                        else
                        "⚠️ استراتژی در این بازه بازده مثبت نداشته است."
                    ),
                    "",
                    "توجه: این بک‌تست شبیه‌سازی تاریخی است و شامل لغزش، کارمزد واقعی، "
                    "عمق دفتر سفارش و تأخیر شبکه به‌صورت کامل نیست."
                ]

                output = "\n".join(report)
                self.root.after(
                    0,
                    lambda o=output: (
                        self.bt_text.delete("1.0", "end"),
                        self.bt_text.insert("1.0", o),
                        self.bt_text.see("1.0"),
                        self.bt_progress.config(text="بک‌تست پایان یافت")
                    )
                )
            except Exception as e:
                logging.exception("backtest worker error: %s", symbol)
                self.root.after(
                    0,
                    lambda err=str(e): (
                        self.bt_text.delete("1.0", "end"),
                        self.bt_text.insert("1.0", f"خطا در بک‌تست:\n{err}"),
                        self.bt_progress.config(text="خطا")
                    )
                )

        try:
            symbol = self.backtest_symbol_combo.get().strip()
            if not symbol:
                sel = self.backtest_symbol_list.curselection()
                if sel:
                    symbol = self.backtest_symbol_list.get(sel[0])
            if not symbol:
                messagebox.showwarning(
                    "بک‌تست", "یک نماد را از فهرست انتخاب کنید."
                )
                return

            try:
                bars = int(self.backtest_bars_entry.get().strip())
            except Exception:
                bars = 2000

            bars = max(500, min(5000, bars))
            threading.Thread(
                target=worker,
                args=(canonical(symbol), bars),
                daemon=True
            ).start()

        except Exception as e:
            messagebox.showerror("خطا", str(e))


    def start(self):
        if getattr(self, "running", False):
            return
        try:
            if hasattr(self, "executor") and self.executor:
                try:
                    self.executor.shutdown(wait=False)
                except:
                    pass
            self.executor = ThreadPoolExecutor(max_workers=max(8, executor_workers))
        except:
            pass
        self.running = True
        try:
            self.lbl_status.config(text="در حال اجرا", fg="#10b981")
        except:
            pass
        try:
            threading.Thread(target=self.main_loop, daemon=True).start()
        except:
            pass
        try:
            say_farsi("اسکالپر شروع شد", True)
        except:
            pass

    def stop(self):
        try:
            self.running = False
            if hasattr(self, "executor") and self.executor:
                try:
                    self.executor.shutdown(wait=False)
                except:
                    pass
            self.lbl_status.config(text="متوقف", fg="#f59e0b")
            say_farsi("اسکالپر متوقف شد", True)
        except:
            pass

    def toggle_talaye_enable(self):
        global TALAYE_ENABLED
        try:
            TALAYE_ENABLED = bool(self.talaye_enabled.get())
            logging.info("TALAYE_ENABLED set to %s", TALAYE_ENABLED)
            try:
                self.update_talaye_label()
            except:
                pass
        except Exception as e:
            logging.debug("toggle_talaye_enable error: %s", e)

    def update_talaye_label(self):
        try:
            txt = f"طلایه آستانه: {TALAYE_MIN_SCORE:.3f}"
            self.lbl_talaye_threshold.config(text=txt)
            if hasattr(self, "talaye_value_label") and self.talaye_value_label:
                self.talaye_value_label.config(text=f"{TALAYE_MIN_SCORE:.3f}")
        except:
            pass

    def on_trailing_toggle(self):
        enabled = bool(self.trailing_enabled.get())
        state = "normal" if enabled else "disabled"
        try:
            for key in ["trailing_percent", "trailing_activate"]:
                b = self._scale_blocks.get(key)
                if b:
                    b.scale.config(state=state)
                    b.btn_plus.config(state=state)
                    b.btn_minus.config(state=state)
        except:
            pass

    def on_trailing_toggle_initial(self):
        try:
            if not self.trailing_enabled.get():
                self.on_trailing_toggle()
        except:
            pass

    def open_rules(self):
        try:
            w = tk.Toplevel(self.root)
            w.title("قوانین")
            w.geometry("1100x600")
            cols = ("rank", "name", "hidden", "condition", "enabled")
            tree = ttk.Treeview(w, columns=cols, show="headings")
            tree.heading("rank", text="رتبه")
            tree.heading("name", text="نام")
            tree.heading("hidden", text="کد داخلی")
            tree.heading("condition", text="شرط")
            tree.heading("enabled", text="وضعیت")
            for c in cols:
                tree.column(c, width=200, anchor="center")
            tree.pack(fill="both", expand=True, padx=10, pady=10)
            tree.tag_configure("on", background="#166534", foreground="white")
            tree.tag_configure("off", background="#991b1b", foreground="white")
            def refresh():
                try:
                    tree.delete(*tree.get_children())
                    for meta in RULES_META:
                        enabled = RULES_ENABLED.get(meta["id"], True)
                        vals = (int(meta.get("rank",0)), meta.get("name",""), meta.get("hidden",""), meta.get("condition_text",""), "فعال" if enabled else "غیرفعال")
                        tree.insert("", "end", values=vals, tags=("on" if enabled else "off",))
                except:
                    pass
            def toggle():
                sel = tree.selection()
                if not sel:
                    messagebox.showinfo("تغییر وضعیت", "یک قانون را انتخاب کنید")
                    return
                rank_val = tree.item(sel[0])["values"][0]
                for meta in RULES_META:
                    if int(meta.get("rank",0)) == int(rank_val):
                        rid = meta["id"]
                        RULES_ENABLED[rid] = not RULES_ENABLED.get(rid, True)
                        refresh()
                        return
            refresh()
            btnf = tk.Frame(w)
            btnf.pack(pady=15)
            tk.Button(btnf, text="تغییر وضعیت قانون انتخابی", command=toggle, bg=CTRL_BTN_BG, fg=CTRL_BTN_FG, width=28).pack(side="left", padx=20)
            tk.Button(btnf, text="بستن", command=w.destroy, bg="#ef4444", fg="white").pack(side="right", padx=20)
        except Exception as e:
            logging.exception("open_rules error: %s", e)

    def build_exe(self):
        try:
            password = simpledialog.askstring("پسورد", "پسورد ساخت EXE را وارد کنید:", show="*")
            if password != "1234":
                messagebox.showerror("خطا", "پسورد اشتباه است!")
                return
            script_path = os.path.abspath(__file__)
            out_dir = os.path.dirname(script_path)
            cmd = ["pyinstaller", "--onefile", "--noconfirm", "--distpath", out_dir, script_path]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                logging.info("[pyinstaller] %s", line.strip())
            p.wait()
            if p.returncode == 0:
                messagebox.showinfo("Build EXE", f"Build finished. Check {out_dir}")
            else:
                messagebox.showerror("Build EXE", f"PyInstaller failed (code {p.returncode})")
        except Exception as e:
            messagebox.showerror("Build EXE", f"Error: {e}")

    def copy_table(self):
        if not pyperclip:
            messagebox.showinfo("کپی", "pyperclip نصب نشده است")
            return
        lines = ["نماد\tقیمت\tتغییر\tRSI\tMACD\tDeM\tBBU\tBBL\tVWAP\tVolK\tSignal\tAdvice\tScore\tProb\tPredMove\tTarget\tMTF\tTag\tSide\tRules"]
        try:
            for iid in self.main_table.get_children():
                lines.append("\t".join(map(str, self.main_table.item(iid)["values"])))
            pyperclip.copy("\n".join(lines))
            messagebox.showinfo("کپی", "جدول کپی شد")
        except Exception as e:
            messagebox.showerror("کپی خطا", str(e))

    def copy_signals(self):
        if not pyperclip:
            messagebox.showinfo("کپی", "pyperclip نصب نشده است")
            return
        lines = ["شناسه\tنماد\tزمان\tورود\tخروج\tزمان خروج\tقیمت فعلی\tسود لحظه‌ای\tنتیجه\tپوزیشن\tامتیاز\tاعتبار\tطلایه\tامتیاز طلایه\tقوانین\tورود دستی\tهدف درصد\tقیمت هدف\tپیش‌بینی(%)\tباقیمانده(%)\tقیمت هدف\tزمان انقضا\tمنقضی"]
        try:
            for iid in self.signals_table.get_children():
                vals = self.signals_table.item(iid)["values"]
                lines.append("\t".join(map(str, vals)))
            pyperclip.copy("\n".join(lines))
            messagebox.showinfo("کپی", "جدول سیگنال‌ها کپی شد")
        except Exception as e:
            messagebox.showerror("کپی خطا", str(e))

    def copy_signals_v2(self):
        if not pyperclip:
            messagebox.showinfo("کپی", "pyperclip نصب نشده است")
            return
        lines = ["شناسه\tنماد\tزمان\tورود\tخروج\tزمان خروج\tقیمت فعلی\tسود لحظه‌ای\tنتیجه\tپوزیشن\tامتیاز\tاعتبار\tطلایه\tامتیاز طلایه\tقوانین\tورود دستی\tهدف درصد\tقیمت هدف\tپیش‌بینی(%)\tباقیمانده(%)\tقیمت هدف\tزمان انقضا\tمنقضی"]
        try:
            for iid in self.signals_v2_table.get_children():
                vals = self.signals_v2_table.item(iid)["values"]
                lines.append("\t".join(map(str, vals)))
            pyperclip.copy("\n".join(lines))
            messagebox.showinfo("کپی", "جدول سیگنال 2 کپی شد")
        except Exception as e:
            messagebox.showerror("کپی خطا", str(e))

    def copy_auto_trade(self):
        if not pyperclip:
            messagebox.showinfo("کپی", "pyperclip نصب نشده است")
            return
        lines = ["زمان\tنماد\tجهت\tمقدار\tارزش\tاهرم\tورود\tقیمت فعلی\tاوج\tتریلینگ استاپ\tحد ضرر\tخروج\tزمان خروج\tسود لحظه‌ای\tسود نهایی\tوضعیت\tنتیجه\tریسک(%)"]
        try:
            for iid in self.autotrade_table.get_children():
                vals = self.autotrade_table.item(iid)["values"]
                lines.append("\t".join(map(str, vals)))
            pyperclip.copy("\n".join(lines))
            messagebox.showinfo("کپی", "جدول اتو ترید کپی شد")
        except Exception as e:
            messagebox.showerror("کپی خطا", str(e))

    def clear_signal_log(self):
        if not messagebox.askyesno("Clear", "آیا مطمئن هستید؟"):
            return
        with state_lock:
            signal_log.clear()
        try:
            self.signals_table.delete(*self.signals_table.get_children())
        except:
            pass
        messagebox.showinfo("پاک شد", "لاگ سیگنال‌ها پاک شد")

    def clear_signal_log_v2(self):
        if not messagebox.askyesno("پاک کردن", "آیا مطمئن هستید؟"):
            return
        with state_lock:
            signal_log_v2.clear()
        try:
            self.signals_v2_table.delete(*self.signals_v2_table.get_children())
        except:
            pass

    def clear_auto_trade_log(self):
        if not messagebox.askyesno("پاک کردن", "آیا مطمئن هستید؟"):
            return
        with state_lock:
            auto_trade_log.clear()
        try:
            self.autotrade_table.delete(*self.autotrade_table.get_children())
        except:
            pass

    def export_signals(self):
        with state_lock:
            has_signals = bool(signal_log)
        if not has_signals:
            messagebox.showinfo("Export", "لاگ سیگنال خالی است.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not p:
            return
        try:
            with open(p, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["id","symbol","time","entry","exit","exit_time","current_price","realized_pct","result","side","score","confidence","talaye","talaye_score","matched_rules","manual_entry","manual_target_pct","manual_target_price","predicted_move_pct","remaining_pct","target_price","expire_time","expired"])
                with state_lock:
                    for r in list(signal_log):
                        w.writerow([
                            r.get("id"), r.get("symbol"), r.get("time_str"), r.get("entry"), r.get("exit"),
                            r.get("exit_time"), r.get("current_price"), r.get("realized_pct"), r.get("result"),
                            r.get("side"), r.get("score"), r.get("confidence", ""), r.get("talaye", False),
                            r.get("talaye_score", 0.0), "|".join(r.get("matched_rules_names", [])),
                            r.get("manual_entry"), r.get("manual_target_pct"), r.get("manual_target_price"),
                            r.get("predicted_move_pct"), r.get("remaining_pct"), r.get("target_price"),
                            r.get("expire_time"), r.get("expired")
                        ])
            messagebox.showinfo("Export", f"Saved to {p}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def on_signal_double_click(self, event):
        try:
            iid = self.signals_table.identify_row(event.y)
            if not iid:
                return
            vals = self.signals_table.item(iid)["values"]
            sid = vals[0]
            rec = None
            with state_lock:
                for r in signal_log:
                    if r.get("id") == sid:
                        rec = r
                        break
            if not rec:
                messagebox.showinfo("خطا", "سیگنال یافت نشد")
                return
            dlg = tk.Toplevel(self.root)
            dlg.title(f"ویرایش سیگنال {rec.get('symbol')}")
            tk.Label(dlg, text=f"نماد: {rec.get('symbol')}").grid(row=0, column=0, columnspan=2, pady=6)
            tk.Label(dlg, text="ورود دستی (قیمت):").grid(row=1, column=0, sticky="e", padx=6, pady=4)
            e_entry = tk.Entry(dlg, width=25)
            e_entry.grid(row=1, column=1, padx=6, pady=4)
            if rec.get("manual_entry") is not None:
                e_entry.insert(0, str(rec.get("manual_entry")))
            tk.Label(dlg, text="هدف (%) :").grid(row=2, column=0, sticky="e", padx=6, pady=4)
            e_pct = tk.Entry(dlg, width=25)
            e_pct.grid(row=2, column=1, padx=6, pady=4)
            if rec.get("manual_target_pct") is not None:
                e_pct.insert(0, str(rec.get("manual_target_pct")))
            tk.Label(dlg, text="پیش‌بینی حرکت (%) :").grid(row=3, column=0, sticky="e", padx=6, pady=4)
            lbl_pred = tk.Label(dlg, text=str(rec.get("predicted_move_pct", "—")), bg="#fff", width=20)
            lbl_pred.grid(row=3, column=1, padx=6, pady=4)
            tk.Label(dlg, text="قیمت هدف :").grid(row=4, column=0, sticky="e", padx=6, pady=4)
            lbl_target = tk.Label(dlg, text=str(rec.get("target_price", "—")), bg="#fff", width=20)
            lbl_target.grid(row=4, column=1, padx=6, pady=4)
            def apply_changes():
                try:
                    me = e_entry.get().strip()
                    mp = e_pct.get().strip()
                    with state_lock:
                        if me:
                            rec["manual_entry"] = float(me)
                        else:
                            rec.pop("manual_entry", None)
                        if mp:
                            rec["manual_target_pct"] = float(mp)
                            entry_price = rec.get("manual_entry") or rec.get("entry")
                            if entry_price:
                                if rec.get("side") == "لانگ":
                                    rec["manual_target_price"] = round(entry_price * (1 + rec["manual_target_pct"]/100.0), 8)
                                else:
                                    rec["manual_target_price"] = round(entry_price * (1 - rec["manual_target_pct"]/100.0), 8)
                        else:
                            rec.pop("manual_target_pct", None)
                            rec.pop("manual_target_price", None)
                    dlg.destroy()
                    self.refresh_tables()
                except Exception as e:
                    messagebox.showerror("خطا", str(e))
            tk.Button(dlg, text="ذخیره", command=apply_changes, bg=CTRL_BTN_BG, fg=CTRL_BTN_FG).grid(row=5, column=0, pady=8)
            tk.Button(dlg, text="انصراف", command=dlg.destroy, bg="#ef4444", fg="white").grid(row=5, column=1, pady=8)
        except Exception as e:
            logging.debug("on_signal_double_click error: %s", e)

    def sort_tree(self, tree, col):
        try:
            reverse = False
            if self.sort_state["tree"] == tree and self.sort_state["col"] == col:
                reverse = not self.sort_state.get("reverse", False)
            treeview_sort_column(tree, col, reverse=reverse)
            self.sort_state = {"tree": tree, "col": col, "reverse": reverse}
        except Exception:
            pass

    def reset_signal_sort(self):
        try:
            tree = self.signals_table
            items = list(tree.get_children())
            def parse_time_str(ts):
                try:
                    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except:
                    return datetime.min
            tuples = []
            for iid in items:
                time_str = tree.set(iid, "time") or tree.item(iid)["values"][2]
                tuples.append((parse_time_str(time_str), iid))
            tuples.sort(reverse=True)
            for index, (_, iid) in enumerate(tuples):
                tree.move(iid, "", index)
            self.sort_state = {"tree": tree, "col": "time", "reverse": True}
        except Exception:
            pass

    def _blink(self):
        try:
            self.blink_state = not self.blink_state
            existing = {self.main_table.item(i)["values"][0]: i for i in self.main_table.get_children()}
            for sym in list(self.talaye_blink_symbols):
                iid = existing.get(sym)
                if not iid:
                    try:
                        self.talaye_blink_symbols.discard(sym)
                    except:
                        pass
                    continue
                tag = "talaye_on" if self.blink_state else "talaye_off"
                self.main_table.item(iid, tags=(tag,))
        except Exception:
            pass
        finally:
            self.root.after(700, self._blink)

    def _refresh_trading_center_summary(self):
        """Refresh compact KPIs for the unified Signal/AI/Auto-Trade dashboard."""
        try:
            with state_lock:
                sigs=list(signal_log)
                orders=list(auto_trade_log)
            open_orders=[o for o in orders if o.get("result") is None]
            strong=sum(1 for r in sigs if str(r.get("side", "")))
            self.lbl_center_signals.config(text=f"سیگنال: {len(sigs)}")
            self.lbl_center_open.config(text=f"معامله باز: {len(open_orders)}")
            ai_count=0
            try:
                ai_count=len(self.ai_table.get_children())
            except Exception:
                pass
            self.lbl_center_ai.config(text=f"AI: {ai_count} نماد")
            if open_orders:
                avg_risk=sum(float(o.get("risk_pct", RISK_PER_TRADE_PCT)) for o in open_orders)/len(open_orders)
                self.lbl_center_risk.config(text=f"Risk میانگین: {avg_risk:.2f}%")
            else:
                self.lbl_center_risk.config(text="Risk: —")
        except Exception:
            pass

    def _build_trade_monitor_tab(self):
        """Create a dedicated, always-readable live monitor for open trades."""
        tab = _make_scrollable_tab(self.notebook, bg="#020617")
        self.trade_monitor_tab = tab
        # Put it directly after the overview tab so it is easy to reach.
        self.notebook.insert(1, tab._scroll_outer, text="🔴 مانیتور معاملات")

        top = tk.Frame(tab, bg="#0f172a")
        top.pack(fill="x", padx=8, pady=(6, 4))
        self.tm_title = tk.Label(top, text="📈 مانیتور زنده معاملات", bg="#0f172a", fg="#e2e8f0",
                                 font=("Tahoma", 14, "bold"))
        self.tm_title.pack(side="left", padx=8)
        self.tm_open = tk.Label(top, text="باز: 0", bg="#0f172a", fg="#22c55e", font=("Tahoma", 11, "bold"))
        self.tm_open.pack(side="left", padx=10)
        self.tm_profit = tk.Label(top, text="سود: 0", bg="#0f172a", fg="#22c55e")
        self.tm_profit.pack(side="left", padx=10)
        self.tm_loss = tk.Label(top, text="ضرر: 0", bg="#0f172a", fg="#ef4444")
        self.tm_loss.pack(side="left", padx=10)
        self.tm_trailing = tk.Label(top, text="Trailing: 0", bg="#0f172a", fg="#38bdf8")
        self.tm_trailing.pack(side="left", padx=10)
        self.tm_clock = tk.Label(top, text="Live", bg="#0f172a", fg="#94a3b8")
        self.tm_clock.pack(side="right", padx=8)

        body = tk.PanedWindow(tab, orient="vertical", sashrelief="raised", bg="#020617", bd=0)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        list_frame = tk.Frame(body, bg="#0f172a", height=270)
        list_frame.grid_propagate(False)
        body.add(list_frame, minsize=210)
        list_frame.grid_rowconfigure(1, weight=1); list_frame.grid_columnconfigure(0, weight=1)
        tk.Label(list_frame, text="معاملات بر اساس اولویت — خطر/نزدیکی به TP/Trailing/زمان", bg="#0f172a",
                 fg="#cbd5e1", anchor="w", font=("Tahoma", 10, "bold")).grid(row=0, column=0, sticky="ew", padx=6, pady=4)

        try:
            style = ttk.Style()
            style.configure("TradeMonitor.Treeview", rowheight=28, font=("Tahoma", 9))
            style.configure("TradeMonitor.Treeview.Heading", font=("Tahoma", 9, "bold"))
        except Exception:
            pass
        cols = ("priority","symbol","side","stage","pnl","entry","current","tp1","tp2","sl","trailing","distance","age")
        self.trade_monitor_table = ttk.Treeview(list_frame, columns=cols, show="headings", height=8, style="TradeMonitor.Treeview")
        titles = {"priority":"اولویت","symbol":"نماد","side":"جهت","stage":"مرحله","pnl":"P/L%",
                  "entry":"ورود","current":"فعلی","tp1":"TP1","tp2":"TP2","sl":"SL",
                  "trailing":"Trailing","distance":"فاصله هدف","age":"سن"}
        widths = {"priority":60,"symbol":105,"side":70,"stage":145,"pnl":80,"entry":105,"current":105,
                  "tp1":105,"tp2":105,"sl":105,"trailing":105,"distance":95,"age":80}
        for c in cols:
            self.trade_monitor_table.heading(c, text=titles[c])
            self.trade_monitor_table.column(c, width=widths[c], anchor="center", stretch=False)
        tv = ttk.Scrollbar(list_frame, orient="vertical", command=self.trade_monitor_table.yview)
        th = ttk.Scrollbar(list_frame, orient="horizontal", command=self.trade_monitor_table.xview)
        self.trade_monitor_table.configure(yscrollcommand=tv.set, xscrollcommand=th.set)
        self.trade_monitor_table.grid(row=1, column=0, sticky="nsew", padx=4)
        tv.grid(row=1, column=1, sticky="ns"); th.grid(row=2, column=0, sticky="ew", padx=4)
        self.trade_monitor_table.tag_configure("profit", background="#14532d", foreground="white")
        self.trade_monitor_table.tag_configure("loss", background="#7f1d1d", foreground="white")
        self.trade_monitor_table.tag_configure("trailing", background="#075985", foreground="white")
        self.trade_monitor_table.tag_configure("danger", background="#9a3412", foreground="white")
        self.trade_monitor_table.tag_configure("normal", background="#1e293b", foreground="#e2e8f0")
        self.trade_monitor_table.bind("<<TreeviewSelect>>", self._on_trade_monitor_select)
        _install_scroll_support(self.trade_monitor_table)

        chart_frame = tk.Frame(body, bg="#020617")
        body.add(chart_frame, minsize=440)
        chart_frame.grid_rowconfigure(1, weight=1); chart_frame.grid_columnconfigure(0, weight=1)
        self.tm_chart_title = tk.Label(chart_frame, text="یک معامله را انتخاب کنید", bg="#020617", fg="#e2e8f0",
                                       font=("Tahoma", 12, "bold"))
        self.tm_chart_title.grid(row=0, column=0, sticky="ew", padx=6, pady=4)
        self.tm_chart_hint = tk.Label(chart_frame, text="Trailing: +1% شروع | مرحله اول: حفظ 30% سود | از +2%: حفظ 50% سود | سپر: کنترل ضرر", bg="#020617", fg="#64748b", font=("Tahoma", 9))
        self.tm_chart_hint.grid(row=2, column=0, sticky="ew", padx=6, pady=(0,4))
        self.trade_chart = tk.Canvas(chart_frame, bg="#020617", highlightthickness=0)
        self.trade_chart.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        # Keep the chart readable on small windows; mouse wheel adjusts the pane sash.
        self.trade_chart.bind("<MouseWheel>", lambda e: self._trade_monitor_wheel(e))
        self.trade_chart.bind("<Configure>", lambda _e: self._draw_trade_chart())
        self._trade_monitor_selected_key = None
        self._trade_monitor_last_orders = []

    def _trade_monitor_wheel(self, event):
        try:
            body = self.trade_monitor_tab.winfo_children()
            # Find the vertical PanedWindow and move its sash so the chart can
            # always be brought into view without losing the table.
            for w in body:
                if isinstance(w, tk.PanedWindow):
                    info = w.sash_coord(0)
                    if info:
                        x, y = info
                        delta = -35 if event.delta > 0 else 35
                        w.sash_place(0, x, max(250, min(w.winfo_height()-300, y+delta)))
                    break
        except Exception:
            pass

    def _trade_key(self, order):
        return (str(order.get("ts", "")), canonical(order.get("symbol", "")), str(order.get("entry_price", "")))

    def _trade_stage(self, o):
        if o.get("result"):
            return f"بسته: {o.get('result')}"
        if o.get("trailing_active"):
            return "🔵 TRAILING فعال"
        if o.get("tp1_hit"):
            return "🟢 TP1 ✓ → TP2"
        try:
            cur=safe_float(o.get("current_price")); entry=safe_float(o.get("entry_price")); tp1=safe_float(o.get("take_profit_1"))
            side=str(o.get("side", "long"))
            if tp1 > 0 and entry > 0:
                if side == "long" and cur >= entry: return "🟡 Entry → TP1"
                if side == "short" and cur <= entry: return "🟡 Entry → TP1"
        except Exception:
            pass
        return "⚪ Entry"

    def _trade_priority(self, o):
        """Higher number = more important to keep at the top."""
        if o.get("result") is not None: return -10
        pnl=safe_float(o.get("live_pct"))
        side=str(o.get("side", "long"))
        cur=safe_float(o.get("current_price")); entry=safe_float(o.get("entry_price"))
        sl=safe_float(o.get("stop_loss_price")); tp1=safe_float(o.get("take_profit_1")); tp2=safe_float(o.get("take_profit_2"))
        if o.get("trailing_active"):
            return 90 + min(10, abs(pnl))
        # Danger zone: price is close to SL in raw-price terms.
        if cur > 0 and sl > 0 and entry > 0:
            span=abs(entry-sl) or entry*0.01
            dist=abs(cur-sl)/span
            if dist <= 0.35: return 100 + max(0, 35-dist*100)
        # Next target is close.
        target=tp1 if not o.get("tp1_hit") else tp2
        if cur > 0 and target > 0:
            span=max(abs(target-entry), entry*0.001)
            d=abs(target-cur)/span
            if d <= 0.35: return 80 + max(0, 35-d*100)
        return 50 + min(20, max(-20, pnl))

    def _on_trade_monitor_select(self, _event=None):
        try:
            sel=self.trade_monitor_table.selection()
            if not sel: return
            vals=self.trade_monitor_table.item(sel[0], "values")
            if not vals: return
            self._trade_monitor_selected_key = vals[0]  # hidden stable key stored in priority field suffix
            # Resolve by visible symbol + entry/current when needed.
            sym=str(vals[1]); entry=safe_float(vals[5])
            with state_lock:
                matches=[o for o in auto_trade_log if canonical(o.get("symbol"))==canonical(sym) and abs(safe_float(o.get("entry_price"))-entry) <= max(entry*1e-10, 1e-12)]
            if matches:
                self._trade_monitor_selected_key=self._trade_key(matches[0])
                self._draw_trade_chart(matches[0])
        except Exception:
            pass

    def _draw_trade_chart(self, selected=None):
        """Render a high-contrast, non-overlapping trade chart with a compact level panel."""
        try:
            c = self.trade_chart
            c.delete("all")
            if selected is None:
                key = self._trade_monitor_selected_key
                with state_lock:
                    orders = list(auto_trade_log)
                if key is not None:
                    selected = next((o for o in orders if self._trade_key(o) == key), None)
                if selected is None:
                    selected = next((o for o in orders if o.get("result") is None), None)
            if not selected:
                w, h = max(700, c.winfo_width()), max(420, c.winfo_height())
                c.create_text(w/2, h/2, text="هیچ معامله‌ای برای نمایش وجود ندارد", fill="#94a3b8",
                              font=("Tahoma", 14, "bold"))
                return

            self._trade_monitor_selected_key = self._trade_key(selected)
            w, h = max(820, c.winfo_width()), max(430, c.winfo_height())
            hist = list(selected.get("price_history") or [])
            entry = safe_float(selected.get("entry_price"))
            cur = safe_float(selected.get("current_price"))
            if not hist:
                hist = [{"ts": selected.get("ts", time.time()), "price": entry},
                        {"ts": time.time(), "price": cur or entry}]
            hist = [x for x in hist if safe_float(x.get("price")) > 0]
            if len(hist) == 1:
                hist.append({"ts": time.time(), "price": cur or safe_float(hist[0].get("price"))})

            levels = [
                ("Entry", entry, "#f59e0b"),
                ("TP1", safe_float(selected.get("take_profit_1")), "#22c55e"),
                ("TP2", safe_float(selected.get("take_profit_2")), "#86efac"),
                ("SL", safe_float(selected.get("stop_loss_price")), "#ef4444"),
                ("Trail", safe_float(selected.get("trailing_stop")), "#38bdf8"),
            ]
            prices = [safe_float(x.get("price")) for x in hist if safe_float(x.get("price")) > 0]
            prices += [v for _, v, _ in levels if v > 0]
            if not prices:
                return
            lo, hi = min(prices), max(prices)
            span = hi - lo
            pad = span * 0.16 if span > 0 else max(abs(entry) * 0.004, 1.0)
            lo -= pad; hi += pad

            # Main plot + right-side level cards.
            left, top, right_panel, bottom = 78, 34, 205, 54
            plot_right = w - right_panel - 22
            plot_bottom = h - bottom
            pw, ph = max(300, plot_right-left), max(220, plot_bottom-top)
            def px(i): return left + (i / max(1, len(hist)-1)) * pw
            def py(v): return top + (hi-v) / max(1e-12, hi-lo) * ph

            # Subtle plot background and glow-like layers.
            c.create_rectangle(left-8, top-8, plot_right+8, plot_bottom+8, fill="#08111f", outline="#172554", width=1)
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                yy = top + ph * frac
                val = hi - (hi-lo) * frac
                c.create_line(left, yy, plot_right, yy, fill="#132238", width=1)
                c.create_text(left-12, yy, text=format_price(val), fill="#7c8da6", anchor="e",
                              font=("Tahoma", 8))
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                xx = left + pw * frac
                c.create_line(xx, top, xx, plot_bottom, fill="#0d1a2b", width=1)

            # Price path + soft filled area.
            pts = [(px(i), py(safe_float(p.get("price")))) for i,p in enumerate(hist)]
            if len(pts) >= 2:
                poly = [(x, plot_bottom) for x,_ in pts] + list(reversed(pts))
                c.create_polygon([z for pt in poly for z in pt], fill="#0b1f33", outline="")
                c.create_line(*[z for pt in pts for z in pt], fill="#60a5fa", width=3, smooth=True)
                # Highlight recent segment.
                recent = pts[max(0, len(pts)-18):]
                if len(recent) >= 2:
                    c.create_line(*[z for pt in recent for z in pt], fill="#a5f3fc", width=4, smooth=True)
            if pts:
                cx, cy = pts[-1]
                c.create_oval(cx-7, cy-7, cx+7, cy+7, fill="#e0f2fe", outline="#38bdf8", width=3)
                c.create_oval(cx-2, cy-2, cx+2, cy+2, fill="#38bdf8", outline="")

            # Level lines; labels live in the dedicated panel so they never overlap the chart.
            for name, value, color in levels:
                if value <= 0: continue
                yy = py(value)
                c.create_line(left, yy, plot_right, yy, fill=color, width=2, dash=(7,4))

            # Current-price badge anchored to the plot, with bounded coordinates.
            if cur > 0:
                cy = max(top+18, min(plot_bottom-18, py(cur)))
                badge_x1, badge_x2 = max(left+8, plot_right-152), plot_right-8
                c.create_rectangle(badge_x1, cy-13, badge_x2, cy+13, fill="#0f2a43", outline="#38bdf8", width=1)
                c.create_text((badge_x1+badge_x2)/2, cy, text=f"LIVE  {format_price(cur)}", fill="#e0f2fe",
                              font=("Tahoma", 9, "bold"))

            # Right-side glass-style info panel.
            panel_x = plot_right + 18
            c.create_rectangle(panel_x, top-8, w-18, plot_bottom+8, fill="#0b1220", outline="#1e293b", width=1)
            side = "LONG" if str(selected.get("side")) == "long" else "SHORT"
            stage = self._trade_stage(selected)
            pnl = safe_float(selected.get("live_pct"))
            pnl_usdt = safe_float(selected.get("pnl_usdt") if selected.get("result") is not None else selected.get("live_pnl_usdt"))
            title = f"{selected.get('symbol','—')}  •  {side}"
            c.create_text(panel_x+12, top+12, text=title, fill="#f8fafc", anchor="w", font=("Tahoma", 11, "bold"))
            c.create_text(panel_x+12, top+34, text=f"مرحله: {stage}", fill="#94a3b8", anchor="w", font=("Tahoma", 9))
            pnl_color = "#22c55e" if pnl_usdt >= 0 else "#f87171"
            c.create_text(panel_x+12, top+63, text=f"P/L  {pnl:+.3f}%  |  {pnl_usdt:+.4f} USDT", fill=pnl_color, anchor="w", font=("Tahoma", 12, "bold"))

            card_y = top + 88
            card_h = 43
            for name, value, color in levels:
                if value <= 0: continue
                c.create_rectangle(panel_x+10, card_y, w-28, card_y+card_h, fill="#111c2d", outline="#1f3148")
                c.create_rectangle(panel_x+10, card_y, panel_x+14, card_y+card_h, fill=color, outline="")
                c.create_text(panel_x+24, card_y+14, text=name, fill="#cbd5e1", anchor="w", font=("Tahoma", 8, "bold"))
                c.create_text(w-38, card_y+14, text=format_price(value), fill="#f8fafc", anchor="e", font=("Tahoma", 9, "bold"))
                dist = ((value-cur)/cur*100) if cur > 0 else 0
                c.create_text(panel_x+24, card_y+30, text=f"فاصله {dist:+.2f}%", fill="#64748b", anchor="w", font=("Tahoma", 7))
                card_y += card_h + 7

            c.create_text(left, h-19, text=f"{len(hist)} نقطه  •  Bid/Ask برای تریگر خروج  •  بروزرسانی زنده",
                          fill="#64748b", anchor="w", font=("Tahoma", 8))
            self.tm_chart_title.config(text=f"{selected.get('symbol','—')}  |  {side}  |  {stage}  |  P/L {pnl:+.3f}% / {pnl_usdt:+.4f} USDT")
        except Exception:
            logging.exception("Trade monitor chart draw failed")

    def update_trade_monitor(self):
        try:
            with state_lock: orders=list(auto_trade_log)
            now=time.time()
            # Open trades first, then completed trades; within open trades use priority.
            ranked=[]
            for o in orders:
                oo=dict(o); oo["_priority"] = self._trade_priority(oo); ranked.append(oo)
            ranked.sort(key=lambda o:(o.get("result") is not None, -o.get("_priority",0), -safe_float(o.get("ts"))))
            self._trade_monitor_last_orders=ranked
            open_orders=[o for o in ranked if o.get("result") is None]
            prof=sum(1 for o in open_orders if safe_float(o.get("live_pct"))>0)
            loss=sum(1 for o in open_orders if safe_float(o.get("live_pct"))<0)
            trail=sum(1 for o in open_orders if o.get("trailing_active"))
            self.tm_open.config(text=f"باز: {len(open_orders)}")
            self.tm_profit.config(text=f"سود: {prof}")
            self.tm_loss.config(text=f"ضرر: {loss}")
            self.tm_trailing.config(text=f"Trailing: {trail}")
            self.tm_clock.config(text=datetime.now().strftime("%H:%M:%S"))
            selected_key=self._trade_monitor_selected_key
            self.trade_monitor_table.delete(*self.trade_monitor_table.get_children())
            for idx,o in enumerate(ranked,1):
                cur=safe_float(o.get("current_price")); entry=safe_float(o.get("entry_price")); pnl=safe_float(o.get("live_pct"))
                side=str(o.get("side","long")); target=safe_float(o.get("take_profit_1")) if not o.get("tp1_hit") else safe_float(o.get("take_profit_2"))
                dist=(abs(target-cur)/cur*100) if target>0 and cur>0 else 0
                age=max(0,int(now-safe_float(o.get("ts")))) if o.get("ts") else 0
                age_txt=f"{age//3600}h" if age>=3600 else (f"{age//60}m" if age>=60 else f"{age}s")
                priority=f"{idx}" + (" 🔥" if idx<=3 and o.get("result") is None else "")
                vals=(priority,o.get("symbol","—"),"LONG" if side=="long" else "SHORT",self._trade_stage(o),f"{pnl:+.3f}",
                      format_price(entry),format_price(cur),format_price(o.get("take_profit_1")) if o.get("take_profit_1") else "—",
                      format_price(o.get("take_profit_2")) if o.get("take_profit_2") else "—",format_price(o.get("stop_loss_price")) if o.get("stop_loss_price") else "—",
                      format_price(o.get("trailing_stop")) if o.get("trailing_stop") else "—",f"{dist:.2f}%",age_txt)
                tag="normal"
                if o.get("trailing_active"): tag="trailing"
                elif pnl>0: tag="profit"
                elif pnl<0: tag="loss"
                sl=safe_float(o.get("stop_loss_price"))
                if not o.get("result") and cur>0 and sl>0 and entry>0:
                    if abs(cur-sl)/max(abs(entry-sl),entry*0.001)<0.25: tag="danger"
                iid=self.trade_monitor_table.insert("","end",values=vals,tags=(tag,))
                # Store the exact key without exposing it visually.
                self.trade_monitor_table.set(iid,"priority",priority)
                if selected_key is not None and self._trade_key(o)==selected_key:
                    self.trade_monitor_table.selection_set(iid)
            self._draw_trade_chart()
        except Exception:
            logging.exception("Trade monitor update failed")

    def update_auto_trade_tab(self):
        try:
            self.set_tab_activity("مرکز معاملات هوشمند", "در حال به‌روزرسانی معاملات", "active")
            try:
                scroll_pos = self.autotrade_table.yview()[0]
            except:
                scroll_pos = 0.0
            try:
                selection = self.autotrade_table.selection()
                sel = selection[0] if selection else None
            except:
                sel = None
            try:
                self.autotrade_table.delete(*self.autotrade_table.get_children())
            except:
                pass
            with state_lock:
                local_orders = list(auto_trade_log)
            if not local_orders:
                self.autotrade_table.insert(
                    "", "end", values=(
                        "—", "—", "—", "—", "—", "—", "—", "—",
                        "—", "—", "—", "—", "—", "—", "—",
                        "—", "—", "در انتظار معامله شبیه‌سازی‌شده", "—", "—"
                    )
                )
            for order in local_orders:
                try:
                    stop_loss_display = f"{order.get('stop_loss_pct', STOP_LOSS_PCT_DEFAULT)}%"
                    try:
                        sl_price = order.get("stop_loss_price")
                        if sl_price is not None:
                            stop_loss_display = f"{order.get('stop_loss_pct', STOP_LOSS_PCT_DEFAULT)}% ({format_price(sl_price)})"
                    except:
                        pass
                    vals = (
                        order["time"], order["symbol"], order["direction"],
                        f"{order['amount']:.8f}", f"{order['usdt_value']:.1f}",
                        f"{order['leverage']}x", format_price(order["entry_price"]),
                        format_price(order["current_price"]),
                        format_price(order.get("peak_price")) if order.get("peak_price") else "—",
                        format_price(order.get("trailing_stop")) if order.get("trailing_stop") else ("فعال" if order.get("trailing_active") else "غیرفعال"),
                        stop_loss_display,
                        format_price(order.get("take_profit_1")) if order.get("take_profit_1") else "—",
                        format_price(order.get("take_profit_2")) if order.get("take_profit_2") else "—",
                        format_price(order["exit_price"]) if order["exit_price"] else "—",
                        order["exit_time"] if order["exit_time"] else "—",
                        f"{order.get('live_pct', 0):+.3f}",
                        f"{order.get('final_pct', 0):+.3f}" if order.get("final_pct") is not None else "—",
                        order["status"], order["result"] or "در جریان",
                        f"{order.get('risk_pct', RISK_PER_TRADE_PCT):.2f}%",
                        "🗑 حذف"
                    )
                    iid = self.autotrade_table.insert("", "end", values=vals)
                    result = order.get("result", "")
                    if result == "تریلینگ" and order.get("final_pct", 0) > 0:
                        self.autotrade_table.item(iid, tags=("trail_profit",))
                    elif result == "تریلینگ":
                        self.autotrade_table.item(iid, tags=("loss",))
                    elif result == "سپر دفاعی" and order.get("slippage", False):
                        self.autotrade_table.item(iid, tags=("stop_loss_slip",))
                    elif result == "سپر دفاعی":
                        self.autotrade_table.item(iid, tags=("stop_loss",))
                    elif result == "درست":
                        self.autotrade_table.item(iid, tags=("win",))
                    elif result == "غلط":
                        self.autotrade_table.item(iid, tags=("loss",))
                    else:
                        self.autotrade_table.item(iid, tags=("open",))
                except:
                    pass
            try:
                if scroll_pos > 0:
                    self.autotrade_table.yview_moveto(scroll_pos)
                if sel and self.autotrade_table.exists(sel):
                    self.autotrade_table.selection_set(sel)
                    self.autotrade_table.see(sel)
            except:
                pass
            self._refresh_trading_center_summary()
            try:
                self.update_trade_monitor()
            except Exception:
                pass
            self.set_tab_activity("مرکز معاملات هوشمند", f"{len(local_orders)} معامله", "done")
        except Exception:
            self.set_tab_activity("مرکز معاملات هوشمند", "خطا در به‌روزرسانی معاملات", "done")

    def _ensure_signal_from_overview(self, r):
        """Create a visible signal record from a completed Overview analysis."""
        try:
            sym=str(r[0]); signal=str(r[10] if len(r)>10 else "")
            score=float(r[12]) if len(r)>12 else 0.0
            if score==0 or signal in ("نگهداری","در حال تحلیل","⚪ WAIT"): return
            side="لانگ" if "خرید" in signal or "BUY" in signal else "شورت" if "فروش" in signal or "SELL" in signal else ""
            if not side: return
            now=time.time()
            with state_lock:
                recent=[x for x in signal_log if x.get("symbol")==sym and now-float(x.get("time",0))<SIGNAL_COOLDOWN]
                if recent: return
                entry=safe_float(valid_symbols_map.get(sym))
                pred=float(r[14]) if len(r)>14 else 0.0
                target=r[15] if len(r)>15 else "—"
                rec={"id":len(signal_log)+1,"symbol":sym,"time":now,"time_str":time.strftime("%Y-%m-%d %H:%M:%S"),"entry":entry,"exit":None,"exit_time":None,"current_price":entry,"live_pct":0.0,"result":"","side":side,"score":score,"remaining_pct":pred,"talaye":bool(r[19]) if len(r)>19 else False,"talaye_score":0.0,"matched_rules_names":[],"manual_entry":None,"manual_target_pct":pred,"manual_target_price":safe_float(target) if isinstance(target,(int,float)) else None,"predicted_move_pct":pred,"target_price":safe_float(target) if isinstance(target,(int,float)) else None,"expire_time":None,"expired":False}
                signal_log.append(rec)
                if len(signal_log)>MAX_SIGNAL_LOG: del signal_log[:-MAX_SIGNAL_LOG]
        except Exception: pass

    def _ensure_signal_from_decision(self, r, source="AI"):
        """Convert a sufficiently confident BUY/SELL decision into the visible signal log.
        This is a display/logging bridge only; it does not place a live order.
        """
        try:
            sym = canonical(r.get("symbol", ""))
            decision = str(r.get("decision", ""))
            conf = float(r.get("confidence", 0.0) or 0.0)
            score = float(r.get("score", 50.0) or 50.0)
            if decision not in {"BUY", "BUY++", "SELL", "SELL++"} or conf < 60.0 or not sym:
                return
            side = "لانگ" if decision.startswith("BUY") else "شورت"
            now = time.time()
            with state_lock:
                for x in signal_log:
                    if x.get("symbol") == sym and x.get("side") == side and now - float(x.get("time", 0)) < SIGNAL_COOLDOWN:
                        return
                entry = safe_float(r.get("entry") or valid_symbols_map.get(sym))
                if entry <= 0:
                    return
                pred = safe_float(r.get("predicted_move_pct")) or safe_float(r.get("target_pct")) or 0.0
                target = safe_float(r.get("tp1")) or safe_float(r.get("target_price"))
                if not target and pred:
                    target = entry * (1 + pred / 100.0) if side == "لانگ" else entry * (1 - pred / 100.0)
                global signal_id_counter
                signal_id_counter += 1
                rec = {
                    "id": signal_id_counter, "symbol": sym, "time": now,
                    "time_str": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                    "entry": entry, "exit": None, "exit_time": None, "current_price": entry,
                    "live_pct": 0.0, "result": "", "side": side, "score": score,
                    "confidence": conf, "remaining_pct": pred, "predicted_move_pct": pred,
                    "target_price": target if target > 0 else None, "manual_entry": None,
                    "manual_target_pct": pred, "manual_target_price": target if target > 0 else None,
                    "talaye": False, "talaye_score": 0.0,
                    "matched_rules_names": [source], "expire_time": None, "expired": False
                }
                signal_log.append(rec)
                _trim_signal_log()
            self.root.after(0, self.refresh_tables)
        except Exception as e:
            logging.debug("_ensure_signal_from_decision failed: %s", e)

    def refresh_tables(self):
        """Schedule signal-table refresh without doing network I/O on Tk's UI thread."""
        try:
            if getattr(self, "_signal_refresh_pending", False):
                return
            self._signal_refresh_pending = True
            self.set_tab_activity("مرکز معاملات هوشمند", "دریافت قیمت‌ها...", "active")
            with state_lock:
                symbols = [rec.get("symbol") for rec in signal_log if rec.get("symbol")]

            def worker(sym_list):
                try:
                    price_map = build_price_map(sym_list)
                except Exception as exc:
                    logging.debug("signal price refresh failed: %s", exc)
                    price_map = {}
                try:
                    self.root.after(0, lambda pm=price_map: self._refresh_tables_ui(pm))
                except Exception:
                    self._signal_refresh_pending = False

            threading.Thread(target=worker, args=(symbols,), daemon=True, name="SignalRefresh").start()
        except Exception:
            self._signal_refresh_pending = False
            self.set_tab_activity("مرکز معاملات هوشمند", "خطا در زمان‌بندی", "done")

    def _refresh_tables_ui(self, price_map):
        try:
            with state_lock:
                for rec in signal_log:
                    try:
                        sym = rec.get("symbol")
                        current = price_map.get(sym)
                        rec["current_price"] = current
                        entry_price = rec.get("manual_entry") if rec.get("manual_entry") is not None else rec.get("entry")
                        if current is not None and entry_price and entry_price != 0:
                            if rec.get("side") == "لانگ":
                                live_pct = (current - entry_price) / entry_price * 100.0
                            else:
                                live_pct = (entry_price - current) / entry_price * 100.0
                            rec["live_pct"] = round(live_pct, 4)
                        else:
                            rec["live_pct"] = None
                        try:
                            pred = rec.get("predicted_move_pct")
                            if pred is not None and entry_price and current:
                                if rec.get("side") == "لانگ":
                                    moved = (current - entry_price) / entry_price * 100.0
                                else:
                                    moved = (entry_price - current) / entry_price * 100.0
                                rem = max(0.0, pred - moved)
                                rec["remaining_pct"] = round(rem, 2)
                            else:
                                rec["remaining_pct"] = rec.get("predicted_move_pct")
                        except:
                            rec["remaining_pct"] = rec.get("predicted_move_pct")
                    except:
                        continue
            lwins = lloss = swins = sloss = 0
            with state_lock:
                for rec in signal_log:
                    if rec.get("result") in ["درست", "غلط"]:
                        side = rec.get("side")
                        if side == "لانگ":
                            if rec["result"] == "درست":
                                lwins += 1
                            else:
                                lloss += 1
                        elif side == "شورت":
                            if rec["result"] == "درست":
                                swins += 1
                            else:
                                sloss += 1
            try:
                self.lbl_long_wins.config(text=f"لانگ‌ها: {lwins} برد / {lloss} باخت")
                self.lbl_short_wins.config(text=f"شورت‌ها: {swins} برد / {sloss} باخت")
            except:
                pass
            with state_lock:
                sorted_signals = sorted(signal_log, key=lambda x: x.get("time", 0), reverse=True)
            try:
                self.signals_table.delete(*self.signals_table.get_children())
            except:
                pass
            if not sorted_signals:
                self.signals_table.insert("", "end", values=(
                    "—", "—", "—", "—", "—", "—", "—", "—",
                    "در انتظار سیگنال معتبر", "—", "—", "—", "—", "—",
                    "—", "—", "—", "—", "—", "—", "—", "—", "خیر"
                ))
            for rec in sorted_signals:
                confidence_display = ""
                try:
                    confidence_display = f"{rec.get('remaining_pct'):.2f}%" if rec.get("remaining_pct") is not None else ""
                except:
                    confidence_display = ""
                vals = (
                    rec.get("id"), rec.get("symbol"), rec.get("time_str"),
                    format_price(rec.get("entry")),
                    format_price(rec.get("exit")) if rec.get("exit") else "",
                    rec.get("exit_time") if rec.get("exit_time") else "",
                    format_price(rec.get("current_price")) if rec.get("current_price") else "",
                    f"{rec.get('live_pct'):+.3f}" if rec.get("live_pct") is not None else "",
                    rec.get("result") if rec.get("result") else "",
                    rec.get("side") if rec.get("side") else "",
                    rec.get("score"), confidence_display,
                    "بله" if rec.get("talaye") else "خیر",
                    round(rec.get("talaye_score", 0.0), 3),
                    "|".join(rec.get("matched_rules_names", [])),
                    format_price(rec.get("manual_entry")) if rec.get("manual_entry") else "",
                    rec.get("manual_target_pct") if rec.get("manual_target_pct") is not None else "",
                    format_price(rec.get("manual_target_price")) if rec.get("manual_target_price") else "",
                    rec.get("predicted_move_pct") if rec.get("predicted_move_pct") is not None else "",
                    f"{rec.get('remaining_pct') if rec.get('remaining_pct') is not None else ''}",
                    format_price(rec.get("target_price")) if rec.get("target_price") else "",
                    rec.get("expire_time") if rec.get("expire_time") else "",
                    "بله" if rec.get("expired") else "خیر",
                    "🔴 فروش فوری / بستن"
                )
                try:
                    iid = self.signals_table.insert("", "end", values=vals)
                    if rec.get("result") == "درست":
                        self.signals_table.item(iid, tags=("win",))
                    elif rec.get("result") == "غلط":
                        self.signals_table.item(iid, tags=("loss",))
                    elif rec.get("expired"):
                        self.signals_table.item(iid, tags=("expired",))
                except:
                    continue
            self._refresh_trading_center_summary()
            self.set_tab_activity("مرکز معاملات هوشمند", f"{len(sorted_signals)} سیگنال", "done")
        except Exception:
            self.set_tab_activity("مرکز معاملات هوشمند", "خطا در به‌روزرسانی", "done")
        finally:
            self._signal_refresh_pending = False

    def refresh_signals_v2(self):
        try:
            lwins = lloss = swins = sloss = 0
            with state_lock:
                for rec in signal_log_v2:
                    if rec.get("result") in ["درست", "غلط"]:
                        side = rec.get("side")
                        if side == "لانگ":
                            if rec["result"] == "درست":
                                lwins += 1
                            else:
                                lloss += 1
                        elif side == "شورت":
                            if rec["result"] == "درست":
                                swins += 1
                            else:
                                sloss += 1
            try:
                self.lbl_long_wins_v2.config(text=f"لانگ‌ها: {lwins} برد / {lloss} باخت")
                self.lbl_short_wins_v2.config(text=f"شورت‌ها: {swins} برد / {sloss} باخت")
            except:
                pass
            with state_lock:
                sorted_signals = sorted(signal_log_v2, key=lambda x: x.get("time", 0), reverse=True)
            try:
                self.signals_v2_table.delete(*self.signals_v2_table.get_children())
            except:
                pass
            if not sorted_signals:
                self.signals_v2_table.insert("", "end", values=(
                    "—", "—", "—", "—", "—", "—", "—", "—",
                    "در انتظار سیگنال معتبر", "—", "—", "—", "—", "—",
                    "—", "—", "—", "—", "—", "—", "—", "—", "خیر"
                ))
            for rec in sorted_signals:
                confidence_display = ""
                try:
                    confidence_display = f"{rec.get('remaining_pct'):.2f}%" if rec.get("remaining_pct") is not None else ""
                except:
                    confidence_display = ""
                vals = (
                    rec.get("id"), rec.get("symbol"), rec.get("time_str"),
                    format_price(rec.get("entry")),
                    format_price(rec.get("exit")) if rec.get("exit") else "",
                    rec.get("exit_time") if rec.get("exit_time") else "",
                    format_price(rec.get("current_price")) if rec.get("current_price") else "",
                    f"{rec.get('live_pct'):+.3f}" if rec.get("live_pct") is not None else "",
                    rec.get("result") if rec.get("result") else "",
                    rec.get("side") if rec.get("side") else "",
                    rec.get("score"), confidence_display,
                    "بله" if rec.get("talaye") else "خیر",
                    round(rec.get("talaye_score", 0.0), 3),
                    "|".join(rec.get("matched_rules_names", [])),
                    format_price(rec.get("manual_entry")) if rec.get("manual_entry") else "",
                    rec.get("manual_target_pct") if rec.get("manual_target_pct") is not None else "",
                    format_price(rec.get("manual_target_price")) if rec.get("manual_target_price") else "",
                    rec.get("predicted_move_pct") if rec.get("predicted_move_pct") is not None else "",
                    f"{rec.get('remaining_pct') if rec.get('remaining_pct') is not None else ''}",
                    format_price(rec.get("target_price")) if rec.get("target_price") else "",
                    rec.get("expire_time") if rec.get("expire_time") else "",
                    "بله" if rec.get("expired") else "خیر"
                )
                try:
                    iid = self.signals_v2_table.insert("", "end", values=vals)
                    if rec.get("result") == "درست":
                        self.signals_v2_table.item(iid, tags=("win",))
                    elif rec.get("result") == "غلط":
                        self.signals_v2_table.item(iid, tags=("loss",))
                    elif rec.get("expired"):
                        self.signals_v2_table.item(iid, tags=("expired",))
                except:
                    continue
        except Exception:
            pass

    def analyze_symbol_with_rules(self, sym: str):
        global signal_id_counter
        sym_key = canonical(sym)

        def empty_result(price_value=None, status="داده در دسترس نیست"):
            ptxt = format_price(price_value) if price_value else "—"
            return (
                sym_key, ptxt, "—", "—", "—", "—", "—", "—", "—", "—",
                status, "نگهداری", 0, 0.0, 0.0, "—", "0%", "neutral",
                "—", False, 0.0, 0.0
            )

        try:
            # market/stats is already mapped, so do not make another orderbook
            # request just to get the current price.
            # FAST OVERVIEW: never fall back to an orderbook request here.
            # market/stats is the authoritative bulk price source; a missing
            # price must not stall the 531-symbol overview with a 1-7s network call.
            price = safe_float(valid_symbols_map.get(sym_key))
            if not price or price <= 0:
                return empty_result(None, "قیمت ناموجود")

            prev = self.data.setdefault(sym_key, {}).get("prev_price")
            change = ((price - prev) / prev * 100.0) if prev and prev > 0 else 0.0
            self.data[sym_key]["prev_price"] = price

            # FAST OVERVIEW: fetch only the primary timeframe for every symbol.
            # Confirmation timeframe is intentionally deferred; requesting 15m +
            # 60m + 4h/ML for all 531 symbols makes the global API gate the
            # dominant bottleneck.
            dfp = get_candles_cached(
                sym_key, PRIMARY_RESOLUTION, n=160,
                valid_symbols_map=valid_symbols_map
            )
            # IMPORTANT: Overview has a strict one-request-per-symbol budget.
            # Do not launch a second deep-history request here when the primary
            # request fails; that was multiplying latency and starving the UI.
            if dfp.empty or len(dfp) < 40:
                return empty_result(price, "کندل کافی نیست")
            dfc = dfp.copy()
            if not OVERVIEW_PRIMARY_ONLY:
                try:
                    dfc2 = get_candles_cached(
                        sym_key, CONFIRM_RESOLUTION, n=80,
                        valid_symbols_map=valid_symbols_map
                    )
                    if not dfc2.empty:
                        dfc = dfc2
                except Exception:
                    pass

            rsi = calculate_rsi(dfp)
            macd = calculate_macd(dfp)
            dem = calculate_demarker(dfp)
            ub, ma, lb = calculate_bollinger_bands(dfp)
            vwap = calculate_vwap_daily(dfp)
            if vwap is None:
                # Fallback VWAP so the column remains useful even when today's
                # candle slice isn't present in a short historical response.
                try:
                    tp = (dfp["high"] + dfp["low"] + dfp["close"]) / 3.0
                    denom = float(dfp["volume"].sum())
                    vwap = float((tp * dfp["volume"]).sum() / denom) if denom > 0 else None
                except Exception:
                    vwap = None

            vol_usdt = float(price) * float(dfp["volume"].tail(10).sum())
            recent_vol_ratio = get_volume_ratio_from_candles(
                dfp, recent_bars=6, prev_bars=48
            )

            # Score deliberately remains interpretable.
            w = load_weights()
            score = 0.0
            if rsi < 42:
                score += 10 * w.get("rsi", 0.3)
            elif rsi > 58:
                score -= 10 * w.get("rsi", 0.3)

            if macd > 0:
                score += 5 * w.get("macd", 0.3)
            elif macd < 0:
                score -= 5 * w.get("macd", 0.3)

            if dem < 0.33:
                score += 7 * w.get("demarker", 0.2)
            elif dem > 0.67:
                score -= 7 * w.get("demarker", 0.2)

            if ub is not None and lb is not None:
                if price < lb:
                    score += 8 * w.get("bollinger", 0.1)
                elif price > ub:
                    score -= 8 * w.get("bollinger", 0.1)

            if vwap:
                if price > vwap:
                    score += 6 * w.get("vwap", 0.1)
                elif price < vwap:
                    score -= 6 * w.get("vwap", 0.1)

            # Confirm with the selected confirmation dataframe. In fast overview
            # mode dfc is the primary dataframe, so this section never adds an
            # extra API request.
            macd_c = calculate_macd(dfc)
            rsi_c = calculate_rsi(dfc)
            confirmed = (
                (rsi < 42 and macd_c > 0 and rsi_c < 55) or
                (rsi > 58 and macd_c < 0 and rsi_c > 45)
            )
            if confirmed:
                score *= 1.20

            adx_value = calculate_adx(dfp)

            # Do not fetch BTC 4h once per symbol. Use an already cached value
            # when available; otherwise leave the overview neutral. A separate
            # background/candidate analysis can populate this cache later.
            market_trend = "neutral"
            if OVERVIEW_ENABLE_BTC_240:
                try:
                    btc_cache = cache.get("overview_btc_240")
                    if btc_cache and time.time() - btc_cache.get("t", 0) < 300:
                        market_trend = str(btc_cache.get("trend", "neutral"))
                except Exception:
                    market_trend = "neutral"

            talaye_applies = False
            talaye_score = 0.0
            try:
                if TALAYE_ENABLED:
                    talaye_score, _ = talaye_siah_eval_proxy(
                        dfp, dfc, price, vol_usdt, sym_key
                    )
                    if (
                        talaye_score >= TALAYE_MIN_SCORE
                        and recent_vol_ratio >= 1.35
                    ):
                        talaye_applies = True
                        score += int(talaye_score * 10 * TALAYE_WEIGHT)
            except Exception:
                talaye_score = 0.0

            if adx_value < 18 and abs(score) >= 8:
                score *= 0.75

            # Don't block a signal solely because BTC's trend endpoint was slow.
            intended_side = "long" if score > 0 else "short"
            if intended_side == "long" and market_trend == "bear":
                score *= 0.65
            elif intended_side == "short" and market_trend == "bull":
                score *= 0.65

            # MTF is optional. Use cached value quickly; calculate only when
            # the symbol is already in the cache.
            cache_key = f"mtf_{sym_key}"
            mtf_align = 0.5
            try:
                cached_mtf = cache.get(cache_key)
                if cached_mtf and time.time() - cached_mtf.get("t", 0) < 300:
                    mtf_align = float(cached_mtf.get("score", 0.5))
            except Exception:
                pass

            matched_rules = []
            try:
                tf = {"1h": recent_vol_ratio}
                matched_rules = evaluate_rules_for_symbol(sym_key, price, dfp, tf)
            except Exception as e:
                logging.debug("rules evaluation error %s: %s", sym_key, e)

            # A small fallback rule set guarantees the signal engine can work
            # even when specialized SMC rules do not trigger.
            rule_ids = [m.get("id") for m in matched_rules]
            if not matched_rules:
                if rsi <= 40 and macd > 0:
                    rule_ids = ["RSI+MACD"]
                elif rsi >= 60 and macd < 0:
                    rule_ids = ["RSI+MACD"]

            pred_move = estimate_predicted_move(
                score, talaye_score, max(1, len(rule_ids)), mtf_align
            )
            if not isinstance(pred_move, (int, float)) or pred_move <= 0:
                pred_move = 0.5

            if self._ml_enabled_state and OVERVIEW_ENABLE_ML:
                # ML is optional enrichment. It is deliberately disabled for
                # the 531-symbol overview pass because model inference may
                # trigger additional history/model work per symbol.
                try:
                    ml_res = predict_with_ensemble(sym_key)
                    prob_est = float(ml_res.get("prob_up", 0.0)) * 100.0
                    pred_ml = ml_res.get("expected_move")
                    if isinstance(pred_ml, (int, float)) and pred_ml > 0:
                        pred_move = float(pred_ml)
                except Exception:
                    prob_est = 50.0
            else:
                prob_est = 50.0

            if score >= 8:
                sig_label = "خرید قوی"
                advice = "خرید قوی"
            elif score >= 5:
                sig_label = "خرید"
                advice = "خرید"
            elif score <= -8:
                sig_label = "فروش قوی"
                advice = "فروش قوی"
            elif score <= -5:
                sig_label = "فروش"
                advice = "فروش"
            else:
                sig_label = "نگهداری"
                advice = "نگهداری"

            if talaye_applies:
                sig_label = "طلایه سیاه - " + ("خرید" if score > 0 else "فروش")
                advice = "خرید قوی (طلایه)" if score > 0 else "فروش قوی (طلایه)"

            tag = (
                "long_strong" if score >= 10 else
                "long" if score >= 5 else
                "short_strong" if score <= -10 else
                "short" if score <= -5 else
                "neutral"
            )
            direction = "لانگ" if score > 0 else "شورت"

            # Signal logging: signal generation now requires either a meaningful
            # score or a Talaye event, not an arbitrary count of specialized rules.
            should_log = (
                abs(score) >= MIN_LOG_SCORE or
                talaye_applies or
                (abs(score) >= 3 and len(rule_ids) >= 1)
            )

            now_ts = time.time()
            sig_signature = _make_signal_signature(score, talaye_applies, rule_ids)
            last_time = last_signal_time_per_symbol.get(sym_key)
            last_sig = last_signal_signature.get(sym_key)
            if should_log and (
                last_time is None
                or (now_ts - last_time) > SIGNAL_COOLDOWN
                or last_sig != sig_signature
            ):
                init_conf = compute_signal_confidence(
                    score, talaye_score, len(rule_ids), mtf_align
                )
                target_price = (
                    price * (1 + pred_move / 100.0)
                    if score >= 0 else
                    price * (1 - pred_move / 100.0)
                )
                with state_lock:
                    signal_id_counter += 1
                    rec = {
                        "id": signal_id_counter,
                        "symbol": sym_key,
                        "time": now_ts,
                        "time_str": datetime.fromtimestamp(now_ts).strftime("%Y-%m-%d %H:%M:%S"),
                        "entry": price,
                        "score": round(float(score), 3),
                        "talaye": talaye_applies,
                        "talaye_score": float(talaye_score),
                        "matched_rules": rule_ids,
                        "matched_rules_names": [
                            m["meta"]["name"] for m in matched_rules
                        ] if matched_rules else list(map(str, rule_ids)),
                        "exit": None,
                        "exit_time": None,
                        "realized_pct": None,
                        "result": None,
                        "side": "لانگ" if score > 0 else "شورت",
                        "expired": False,
                        "expire_time": None,
                        "manual_entry": None,
                        "manual_target_pct": None,
                        "manual_target_price": None,
                        "current_price": price,
                        "created_at": now_ts,
                        "last_update": now_ts,
                        "confidence": init_conf,
                        "probability": prob_est,
                        "prob_up": prob_est,
                        "predicted_move_pct": pred_move,
                        "target_price": target_price,
                        "remaining_pct": pred_move,
                        "mtf_align": mtf_align,
                        "model_conf": 0.0,
                    }
                    signal_log.append(rec)
                    _trim_signal_log()
                    logging.info(
                        "[SIGNAL] %s %s score=%.2f conf=%.1f%% pred=%.2f%%",
                        "BUY" if score > 0 else "SELL", sym_key, float(score), float(init_conf), float(pred_move)
                    )
                    last_signal_time_per_symbol[sym_key] = now_ts
                    last_signal_signature[sym_key] = sig_signature

                # AUTO TRADE: create a SIM order from the same confirmed signal.
                # Real exchange execution remains disabled unless LIVE_TRADING_ENABLED
                # is explicitly enabled elsewhere. We require a stronger signal than
                # the normal signal-log threshold so weak BUY/SELL rows do not trade.
                if (
                    is_auto_trade_enabled()
                    and not should_pause_trading()
                    and (abs(score) >= 8 or talaye_applies)
                    and float(init_conf) >= 72.0
                ):
                    try:
                        place_order(
                            sym_key,
                            "long" if score > 0 else "short",
                            plan={
                                "sl": price * (1 - get_stop_loss_pct() / 100.0) if score > 0 else price * (1 + get_stop_loss_pct() / 100.0),
                                "tp1": target_price,
                                "tp2": price * (1 + 2.0 * pred_move / 100.0) if score > 0 else price * (1 - 2.0 * pred_move / 100.0),
                                "rr": 2.0,
                            },
                            score=score,
                            confidence=init_conf,
                        )
                    except Exception as e:
                        logging.debug("auto trade placement error %s: %s", sym_key, e)

            target = (
                price * (1 + pred_move / 100.0)
                if score >= 0 else
                price * (1 - pred_move / 100.0)
            )
            return (
                sym_key,
                format_price(price),
                f"{change:+.2f}%" if prev else "0.00%",
                round(rsi, 1),
                round(macd, 6),
                round(dem, 3),
                format_price(ub) if ub is not None else "—",
                format_price(lb) if lb is not None else "—",
                format_price(vwap) if vwap is not None else "—",
                f"{vol_usdt / 1000:,.0f}K",
                sig_label,
                advice,
                round(float(score), 2),
                round(float(prob_est), 1),
                round(float(pred_move), 2),
                format_price(target),
                f"{mtf_align:.0%}",
                tag,
                direction,
                bool(talaye_applies),
                float(pred_move),
                float(target),
            )

        except Exception as e:
            logging.exception("analyze_symbol error %s: %s", sym, e)
            return empty_result(price if "price" in locals() else None, f"خطا: {type(e).__name__}")


    def analyze_symbol_with_rules_v2(self, sym: str):
        global signal_id_counter_v2
        sym_key = canonical(sym)
        try:
            price = safe_float(valid_symbols_map.get(sym_key))
            if price <= 0:
                price = get_price_from_orderbook(sym_key, valid_symbols_map)
            if not price:
                return

            df = get_candles_cached(
                sym_key, PRIMARY_RESOLUTION, n=140,
                valid_symbols_map=valid_symbols_map
            )
            if df.empty or len(df) < 40:
                return

            rsi = calculate_rsi(df)
            macd = calculate_macd(df)
            adx = calculate_adx(df)
            vol_ratio = get_volume_ratio_from_candles(
                df, recent_bars=6, prev_bars=48
            )

            score = 0.0
            if rsi < 40:
                score += 5
            elif rsi > 60:
                score -= 5

            if macd > 0:
                score += 4
            elif macd < 0:
                score -= 4

            if adx >= 20:
                score += 2 if score > 0 else -2 if score < 0 else 0

            if vol_ratio >= 1.25:
                score += 2 if score > 0 else -2 if score < 0 else 0

            talaye_score = 0.0
            talaye_applies = False
            try:
                dfc = get_candles_cached(
                    sym_key, CONFIRM_RESOLUTION, n=70,
                    valid_symbols_map=valid_symbols_map
                )
                if not dfc.empty and TALAYE_ENABLED:
                    talaye_score, _ = talaye_siah_eval_proxy(
                        df, dfc, price, price * float(df["volume"].tail(10).sum()), sym_key
                    )
                    if talaye_score >= TALAYE_MIN_SCORE_V2:
                        talaye_applies = True
                        score += 3
            except Exception:
                pass

            direction = "لانگ" if score > 0 else "شورت" if score < 0 else "بی‌طرف"
            # Signal 2 requires one strong independent confirmation, not three
            # specialized SMC rules.
            strong_enough = (
                abs(score) >= 8 or
                talaye_applies or
                (abs(score) >= 6 and vol_ratio >= 1.15)
            )
            if not strong_enough:
                return

            matched_names = []
            if rsi < 40 or rsi > 60:
                matched_names.append("RSI")
            if macd != 0:
                matched_names.append("MACD")
            if adx >= 20:
                matched_names.append("روند ADX")
            if vol_ratio >= 1.25:
                matched_names.append("افزایش حجم")
            if talaye_applies:
                matched_names.append("طلایه")

            now_ts = time.time()
            sig_signature = _make_signal_signature(
                score, talaye_applies, matched_names
            )
            last_time = last_signal_time_per_symbol.get("v2_" + sym_key)
            last_sig = last_signal_signature.get("v2_" + sym_key)
            if (
                last_time is not None
                and (now_ts - last_time) <= SIGNAL_COOLDOWN
                and last_sig == sig_signature
            ):
                return

            mtf = 0.5
            try:
                ck = f"mtf_{sym_key}"
                if ck in cache:
                    mtf = float(cache[ck].get("score", 0.5))
            except Exception:
                pass

            pred_move = max(
                0.4,
                float(estimate_predicted_move(
                    score, talaye_score, len(matched_names), mtf
                ))
            )
            prob = min(
                95.0,
                max(
                    5.0,
                    50.0 + score * 4.0 + (mtf - 0.5) * 20.0
                )
            )

            target = (
                price * (1 + pred_move / 100.0)
                if score > 0 else
                price * (1 - pred_move / 100.0)
            )

            with state_lock:
                signal_id_counter_v2 += 1
                rec = {
                    "id": signal_id_counter_v2,
                    "symbol": sym_key,
                    "time": now_ts,
                    "time_str": datetime.fromtimestamp(now_ts).strftime("%Y-%m-%d %H:%M:%S"),
                    "entry": price,
                    "score": round(score, 2),
                    "talaye": talaye_applies,
                    "talaye_score": round(talaye_score, 3),
                    "matched_rules": matched_names,
                    "matched_rules_names": matched_names,
                    "exit": None,
                    "exit_time": None,
                    "live_pct": None,
                    "result": None,
                    "side": direction,
                    "current_price": price,
                    "expired": False,
                    "expire_time": None,
                    "created_at": now_ts,
                    "last_update": now_ts,
                    "confidence": min(99.0, max(1.0, 55 + abs(score) * 4)),
                    "probability": prob,
                    "prob_up": prob,
                    "predicted_move_pct": pred_move,
                    "target_price": target,
                    "remaining_pct": pred_move,
                    "mtf_align": mtf,
                    "model_conf": 0.0,
                }
                signal_log_v2.append(rec)
                # The original code accidentally trimmed the v1 log here.
                if len(signal_log_v2) > MAX_SIGNAL_LOG:
                    del signal_log_v2[:-MAX_SIGNAL_LOG]
                last_signal_time_per_symbol["v2_" + sym_key] = now_ts
                last_signal_signature["v2_" + sym_key] = sig_signature

            try:
                self.root.after(0, self.refresh_signals_v2)
            except Exception:
                pass

        except Exception as e:
            logging.debug("analyze_symbol_v2 error %s: %s", sym_key, e)


    def get_market_trend(self) -> str:
        try:
            df = get_candles_cached("BTCUSDT", 240, n=100, valid_symbols_map=valid_symbols_map)
            if df.empty or len(df) < 50:
                return "neutral"
            sma50 = df["close"].rolling(50).mean().iloc[-1]
            price = df["close"].iloc[-1]
            if price > sma50 * 1.005:
                return "bull"
            elif price < sma50 * 0.995:
                return "bear"
            return "neutral"
        except Exception:
            return "neutral"

    def main_loop(self):
        """Rotating, incremental scanner so the Overview never stays blank."""
        # Process the complete symbol universe in rotating batches.
        # Eight workers are intentionally used here: the history API itself is
        # paced by _udf_gate, so increasing workers does not create API bursts,
        # but it prevents the GUI from appearing to be limited to the first
        # four symbols. Every symbol remains in the rotation.
        batch_size = max(16, int(OVERVIEW_BATCH_SIZE))
        cursor = 0

        while self.running:
            cycle_started = time.time()
            try:
                if should_pause_trading():
                    try:
                        self.lbl_status.config(
                            text="توقف (افت زیاد)",
                            fg="#ef4444"
                        )
                    except Exception:
                        pass
                    time.sleep(min(INTERVAL, 5))
                    continue

                # Always repair/refresh the symbol list from the latest market map.
                active_symbols = [
                    s for s in list(SYMBOLS)
                    if s not in pruned_symbols
                ]
                if not active_symbols:
                    active_symbols = [
                        canonical(s) for s in USER_SYMBOLS
                    ]

                if not active_symbols:
                    try:
                        self.lbl_status.config(
                            text="هیچ نماد فعالی پیدا نشد",
                            fg="#ef4444"
                        )
                    except Exception:
                        pass
                    time.sleep(min(INTERVAL, 5))
                    continue

                if cursor >= len(active_symbols):
                    cursor = 0
                batch = active_symbols[cursor:cursor + batch_size]
                if not batch:
                    cursor = 0
                    batch = active_symbols[:batch_size]
                cursor = (cursor + len(batch)) % max(1, len(active_symbols))

                futures = []
                for s in batch:
                    try:
                        futures.append(
                            self.executor.submit(
                                self.analyze_symbol_with_rules, s
                            )
                        )
                    except Exception as e:
                        logging.debug("submit analysis failed %s: %s", s, e)

                completed = 0
                signals = 0
                batch_rows = []
                deadline = time.time() + max(12, min(20, INTERVAL + 5))
                try:
                    for fut in as_completed(futures, timeout=max(1, deadline - time.time())):
                        try:
                            r = fut.result()
                            if r:
                                completed += 1
                                batch_rows.append(r)
                                try:
                                    if len(r) > 12 and abs(float(r[12])) >= 5:
                                        signals += 1
                                except Exception:
                                    pass
                        except Exception as e:
                            logging.debug("analysis future failed: %s", e)
                except TimeoutError:
                    pass
                except Exception as e:
                    logging.debug("batch collection error: %s", e)
                # One Tk callback per batch instead of one callback per symbol.
                if batch_rows:
                    try:
                        self.root.after(0, lambda rows=batch_rows: self.update_main_table(rows))
                    except Exception:
                        pass

                elapsed = time.time() - cycle_started
                try:
                    self.lbl_status.config(
                        text=(
                            f"فعال | داده: {completed}/{len(batch)} | "
                            f"سیگنال: {signals} | {elapsed:.1f}s"
                        ),
                        fg="#10b981" if completed else "#f59e0b"
                    )
                except Exception:
                    pass

            except Exception as e:
                logging.exception("main_loop error: %s", e)
                try:
                    self.lbl_status.config(
                        text=f"خطا در حلقه اصلی: {e}",
                        fg="#ef4444"
                    )
                except Exception:
                    pass

            # Small pause, then immediately continue. This prevents a long
            # 30-second dead period when one API batch is slow.
            time.sleep(1.0)



    def update_main_table(self, res: List[Tuple]):
        """Update Overview using ONE canonical 20-column row schema.

        Accepted analysis tuple (internal) may contain extra bookkeeping fields
        at positions 19-21. Those fields are never sent to Treeview.
        """
        try:
            y_pos = self.main_table.yview()[0]
            x_pos = self.main_table.xview()[0]
        except Exception:
            y_pos, x_pos = 0.0, 0.0

        try:
            existing = {}
            for iid in self.main_table.get_children():
                v = self.main_table.item(iid).get("values", [])
                if v:
                    existing[str(v[0])] = iid
        except Exception:
            existing = {}

        def _num(v, default=0.0):
            try:
                return float(v)
            except Exception:
                return default

        for r in (res or []):
            try:
                self._ensure_signal_from_overview(r)
                if not r:
                    continue
                sym = str(r[0])

                # Internal result layout from analyze_symbol_with_rules:
                # 0..18 are display fields, 19=talaye flag, 20=pred_move raw,
                # 21=target raw. Treeview must receive ONLY 0..18 + rules.
                tag = str(r[17]) if len(r) > 17 and r[17] else "neutral"
                side = str(r[18]) if len(r) > 18 and r[18] else "—"

                with state_lock:
                    latest_rules = [x for x in signal_log if x.get("symbol") == sym]
                rules_list = ""
                if latest_rules:
                    rr = latest_rules[-1]
                    names = rr.get("matched_rules_names") or rr.get("matched_rules") or []
                    rules_list = ",".join(map(str, names))

                talaye = bool(r[19]) if len(r) > 19 else False
                if talaye:
                    try:
                        self.talaye_blink_symbols.add(sym)
                    except Exception:
                        pass
                    tag_to_apply = "talaye_on" if getattr(self, "blink_state", False) else "talaye_off"
                else:
                    try:
                        self.talaye_blink_symbols.discard(sym)
                    except Exception:
                        pass
                    tag_to_apply = tag if tag in {"long_strong","long","short_strong","short","neutral"} else "neutral"

                # Fixed, auditable display conversions.
                prob_raw = r[13] if len(r) > 13 else 0.0
                pred_raw = r[14] if len(r) > 14 else 0.0
                prob_display = f"{_num(prob_raw):.1f}%" if isinstance(prob_raw, (int,float)) or str(prob_raw).replace('.','',1).isdigit() else str(prob_raw)
                pred_display = f"{_num(pred_raw):.2f}%" if isinstance(pred_raw, (int,float)) or str(pred_raw).replace('.','',1).replace('-','',1).isdigit() else str(pred_raw)
                target_display = r[15] if len(r) > 15 and r[15] not in (None, "") else "—"
                mtf_display = r[16] if len(r) > 16 else "0%"

                signal_display = str(r[10]) if len(r) > 10 and r[10] else "نگهداری"
                if "خرید قوی" in signal_display:
                    signal_display = "🟢 BUY++ | " + signal_display
                elif "خرید" in signal_display:
                    signal_display = "🟩 BUY | " + signal_display
                elif "فروش قوی" in signal_display:
                    signal_display = "🔴 SELL++ | " + signal_display
                elif "فروش" in signal_display:
                    signal_display = "🟥 SELL | " + signal_display
                elif "نگهداری" in signal_display or "در حال تحلیل" in signal_display:
                    signal_display = "⚪ WAIT"

                # THE ONLY TREEVIEW ROW SHAPE: exactly 20 values.
                vals = (
                    sym,
                    r[1] if len(r) > 1 else "—",
                    r[2] if len(r) > 2 else "—",
                    r[3] if len(r) > 3 else "—",
                    r[4] if len(r) > 4 else "—",
                    r[5] if len(r) > 5 else "—",
                    r[6] if len(r) > 6 else "—",
                    r[7] if len(r) > 7 else "—",
                    r[8] if len(r) > 8 else "—",
                    r[9] if len(r) > 9 else "—",
                    signal_display,
                    r[11] if len(r) > 11 else "نگهداری",
                    r[12] if len(r) > 12 else 0,
                    prob_display,
                    pred_display,
                    target_display,
                    mtf_display,
                    tag,
                    side,
                    rules_list,
                )
                if len(vals) != 20:
                    raise ValueError(f"Overview row schema error: {len(vals)} != 20")

                if sym in existing:
                    iid = existing[sym]
                    self.main_table.item(iid, values=vals, tags=(tag_to_apply,))
                else:
                    iid = self.main_table.insert("", "end", values=vals, tags=(tag_to_apply,))
                    existing[sym] = iid
            except Exception as e:
                logging.debug("update_main_table row failed: %s", e)

        try:
            self.main_table.yview_moveto(y_pos)
            self.main_table.xview_moveto(x_pos)
        except Exception:
            pass

# ---------------- Main Entry ----------------

# ==================== NOBITEX AUTHENTICATED API ====================
# Supports:
#   A) Legacy token authentication: Authorization: Token <token>
#   B) New API Key authentication: Nobitex-Key / Nobitex-Signature / Nobitex-Timestamp
#
# IMPORTANT: The new API Key private key is used locally only to sign requests.
# It must NEVER be sent as a header or request body.

NOBITEX_API_CONFIG_FILE = str(Path(__file__).resolve().parent / "nobitex_api_config.json")
_nobitex_token = ""
_nobitex_api_key = ""
_nobitex_private_key = ""
_nobitex_auth_mode = "token"


def _nobitex_safe_error_body(response):
    """Return useful JSON/text error details without leaking credentials."""
    try:
        data = response.json()
        if isinstance(data, dict):
            code = data.get("code")
            message = data.get("message") or data.get("detail")
            if code or message:
                parts = []
                if code:
                    parts.append(f"code={code}")
                if message:
                    parts.append(str(message))
                return " | ".join(parts)
            return json.dumps(data, ensure_ascii=False)[:500]
        return str(data)[:500]
    except Exception:
        return str(getattr(response, "text", ""))[:500]


def _save_nobitex_config(
    token: str = "",
    api_key: str = "",
    private_key: str = "",
    mode: str = "token",
):
    global _nobitex_token, _nobitex_api_key, _nobitex_private_key, _nobitex_auth_mode
    try:
        _nobitex_token = str(token or "").strip()
        _nobitex_api_key = str(api_key or "").strip()
        _nobitex_private_key = str(private_key or "").strip()
        _nobitex_auth_mode = "api_key" if str(mode).strip().lower() == "api_key" else "token"

        cfg = {
            "token": _nobitex_token,
            "api_key": _nobitex_api_key,
            "private_key": _nobitex_private_key,
            "mode": _nobitex_auth_mode,
        }
        with open(NOBITEX_API_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning("Failed to save Nobitex API config: %s", e)


def _load_nobitex_config():
    global _nobitex_token, _nobitex_api_key, _nobitex_private_key, _nobitex_auth_mode
    try:
        if os.path.exists(NOBITEX_API_CONFIG_FILE):
            with open(NOBITEX_API_CONFIG_FILE, "r", encoding="utf-8") as f:
                j = json.load(f)

            _nobitex_token = str(j.get("token", "") or "").strip()
            _nobitex_api_key = str(
                j.get("api_key", j.get("key", "")) or ""
            ).strip()
            _nobitex_private_key = str(
                j.get("private_key", j.get("privateKey", "")) or ""
            ).strip()

            mode = str(j.get("mode", "") or "").strip().lower()
            if mode in {"api_key", "apikey", "key"}:
                _nobitex_auth_mode = "api_key"
            elif mode == "token":
                _nobitex_auth_mode = "token"
            else:
                _nobitex_auth_mode = (
                    "api_key"
                    if _nobitex_api_key and _nobitex_private_key
                    else "token"
                )
    except Exception as e:
        logging.warning("Failed to load Nobitex API config: %s", e)

    return _nobitex_token


def _nobitex_normalize_private_key(value: str):
    """
    Decode a Nobitex Ed25519 private key from URL-safe base64.
    The docs specify a base64-encoded raw Ed25519 private key.
    """
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Private Key خالی است.")

    # Support users who may have copied the value with whitespace.
    raw = re.sub(r"\s+", "", raw)

    # Add missing base64 padding.
    raw += "=" * (-len(raw) % 4)

    try:
        key_bytes = base64.urlsafe_b64decode(raw.encode("ascii"))
    except Exception as e:
        raise ValueError(f"Private Key معتبر نیست (Base64): {e}")

    if len(key_bytes) != 32:
        raise ValueError(
            f"طول Private Key نامعتبر است: {len(key_bytes)} بایت؛ "
            "برای Ed25519 باید 32 بایت باشد."
        )
    return key_bytes


def _nobitex_ed25519_sign(private_key_b64: str, message: bytes) -> str:
    """
    Sign bytes using Ed25519 and return URL-safe Base64 signature.
    Supports cryptography first, then PyNaCl as fallback.
    """
    key_bytes = _nobitex_normalize_private_key(private_key_b64)

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        private_key = Ed25519PrivateKey.from_private_bytes(key_bytes)
        signature = private_key.sign(message)
    except ImportError:
        try:
            from nacl.signing import SigningKey
            signature = SigningKey(key_bytes).sign(message).signature
        except ImportError:
            raise RuntimeError(
                "برای API Key جدید، کتابخانه cryptography یا PyNaCl نصب نیست. "
                "در محیط Python برنامه اجرا کنید: pip install cryptography"
            )
    except Exception as e:
        raise RuntimeError(f"امضای Ed25519 ناموفق بود: {e}")

    return base64.urlsafe_b64encode(signature).decode("ascii")


def _nobitex_auth_headers(
    method: str = "GET",
    path: str = "",
    raw_body: bytes = b"",
    query_string: str = "",
):
    """
    Build the correct headers for either legacy Token auth or new signed API Key auth.

    Nobitex API Key signing payload:
        timestamp + METHOD + full_path + raw_body
    where full_path includes query string.
    """
    global _nobitex_auth_mode
    _load_nobitex_config()

    mode = _nobitex_auth_mode
    if mode == "api_key" and _nobitex_api_key and _nobitex_private_key:
        import time as _time

        timestamp = str(int(_time.time()))
        full_path = path or "/"
        if query_string:
            full_path += "?" + query_string

        signing_payload = (
            f"{timestamp}{str(method).upper()}{full_path}"
        ).encode("utf-8") + (raw_body or b"")

        signature = _nobitex_ed25519_sign(
            _nobitex_private_key,
            signing_payload
        )

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Nobitex-Key": _nobitex_api_key,
            "Nobitex-Signature": signature,
            "Nobitex-Timestamp": timestamp,
        }
        headers.update(HEADERS)
        return headers

    token = str(_nobitex_token or "").strip()
    if not token:
        raise RuntimeError(
            "هیچ اعتبارنامه‌ای برای Nobitex ثبت نشده است. "
            "در تب API، توکن یا API Key را وارد و ذخیره کنید."
        )

    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    headers.update(HEADERS)
    return headers


# Backward-compatible name used elsewhere in the program.
def nobitex_auth_headers(
    method: str = "GET",
    path: str = "",
    raw_body: bytes = b"",
    query_string: str = "",
):
    return _nobitex_auth_headers(method, path, raw_body, query_string)


def _nobitex_build_query(params):
    from urllib.parse import urlencode
    if not params:
        return ""
    # doseq=True mirrors standard query serialization for simple list params.
    return urlencode(params, doseq=True)


def _nobitex_request(
    method: str,
    path: str,
    *,
    params=None,
    payload=None,
    timeout=None,
    extra_headers=None,
):
    """
    Centralized Nobitex request function.
    Crucially, API-Key POST bodies are serialized once and the exact bytes
    sent over the wire are the bytes used for the Ed25519 signature.
    """
    method = str(method).upper()
    timeout = timeout or REQUEST_TIMEOUT
    query_string = _nobitex_build_query(params)
    full_path = path + (f"?{query_string}" if query_string else "")

    raw_body = b""
    if payload is not None:
        raw_body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    if path == "/auth/login/":
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        headers.update(HEADERS)
    else:
        headers = nobitex_auth_headers(
            method=method,
            path=path,
            raw_body=raw_body,
            query_string=query_string,
        )

    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items() if v is not None})

    url = f"{API_BASE}{full_path}"

    try:
        response = get_session().request(
            method,
            url,
            headers=headers,
            data=raw_body if payload is not None else None,
            timeout=timeout,
        )
    except requests.RequestException as e:
        return False, f"خطای ارتباط با Nobitex: {e}"

    if response.status_code in (200, 201):
        try:
            data = response.json()
            if isinstance(data, dict) and str(data.get("status", "")).lower() in {"failed", "error", "rejected"}:
                return False, data
            return True, data
        except Exception:
            return False, f"پاسخ JSON نامعتبر از Nobitex: {response.text[:500]}"

    body = _nobitex_safe_error_body(response)
    if response.status_code == 401:
        return False, (
            f"HTTP 401 Unauthorized: {body or 'احراز هویت رد شد.'}\n"
            "روش احراز هویت یا اعتبار کلید را بررسی کنید."
        )
    if response.status_code == 403:
        return False, (
            f"HTTP 403 Forbidden: {body or 'دسترسی مجاز نیست.'}\n"
            "در API Key جدید، سطح دسترسی لازم (مثلاً READ یا TRADE) را بررسی کنید."
        )
    if response.status_code == 429:
        return False, f"HTTP 429 Too Many Requests: {body}"
    return False, f"HTTP {response.status_code}: {body}"


def nobitex_login(username: str, password: str, otp: str = ""):
    try:
        payload = {"username": username, "password": password}
        if otp:
            payload["otp"] = otp

        ok, result = _nobitex_request(
            "POST",
            "/auth/login/",
            payload=payload,
        )
        if not ok:
            return False, result

        if isinstance(result, dict) and result.get("key"):
            global _nobitex_token, _nobitex_auth_mode
            _nobitex_token = str(result["key"]).strip()
            _nobitex_auth_mode = "token"
            _save_nobitex_config(
                token=_nobitex_token,
                api_key=_nobitex_api_key,
                private_key=_nobitex_private_key,
                mode="token",
            )
            return True, _nobitex_token

        if isinstance(result, dict) and result.get("non_field_errors"):
            return False, str(result.get("non_field_errors"))

        return False, f"پاسخ ورود نامشخص: {result}"
    except Exception as e:
        return False, str(e)


def nobitex_get_profile():
    return _nobitex_request("GET", "/users/profile")


def nobitex_get_wallets():
    return _nobitex_request("GET", "/users/wallets/list")

def nobitex_get_wallets_by_type(wallet_type: str = "spot"):
    """Return Nobitex wallet list for a specific wallet type (spot/margin).

    The v2 wallets endpoint supports type=spot|margin.  We keep the existing
    /users/wallets/list call as a fallback because it is widely supported.
    """
    wt = str(wallet_type or "spot").strip().lower()
    if wt not in {"spot", "margin"}:
        wt = "spot"
    ok, res = _nobitex_request("GET", "/v2/wallets", params={"type": wt})
    if ok:
        # v2 may return wallets as a dict keyed by currency. Normalize it to
        # the same list shape used by /users/wallets/list.
        if isinstance(res, dict) and isinstance(res.get("wallets"), dict):
            rows = []
            for cur, data in res.get("wallets", {}).items():
                item = dict(data or {})
                item.setdefault("currency", str(cur).lower())
                item.setdefault("blockedBalance", item.get("blocked", "0"))
                item.setdefault("activeBalance", item.get("balance", "0"))
                rows.append(item)
            return True, {"status": res.get("status", "ok"), "wallets": rows, "type": wt}
        return True, res
    return False, res


def _nobitex_symbol_parts(symbol: str):
    sym = normalize_symbol(symbol)
    if not sym:
        raise ValueError("نماد بازار خالی است.")

    if sym.endswith("USDT"):
        base = sym[:-4]
        if not base:
            raise ValueError(f"نماد نامعتبر: {symbol}")
        return base.lower(), "usdt"

    if sym.endswith("IRT"):
        base = sym[:-3]
        if not base:
            raise ValueError(f"نماد نامعتبر: {symbol}")
        return base.lower(), "rls"

    # Accept RLS spelling too.
    if sym.endswith("RLS"):
        base = sym[:-3]
        if not base:
            raise ValueError(f"نماد نامعتبر: {symbol}")
        return base.lower(), "rls"

    # Conservative fallback: treat a bare symbol as an IRT market.
    return sym.lower(), "rls"


def nobitex_get_margin_markets():
    return _nobitex_request("GET", "/margin/markets/list")


def nobitex_get_margin_delegation_limit(market: str):
    return _nobitex_request("GET", "/margin/v2/delegation-limit", params={"market": canonical(market)})


def nobitex_get_active_positions(symbol: str = ""):
    params = {"status": "active"}
    if symbol:
        base_cur, dst_cur = _nobitex_symbol_parts(symbol)
        params["srcCurrency"] = base_cur
        params["dstCurrency"] = dst_cur
    return _nobitex_request("GET", "/positions/list", params=params)


def nobitex_get_position_status(position_id):
    return _nobitex_request("GET", f"/positions/{int(position_id)}/status")


def nobitex_close_margin_position(position_id, amount: float, order_type: str = "market", price: float = 0.0):
    payload = {"amount": str(amount), "execution": str(order_type).lower()}
    if str(order_type).lower() != "market" and price and price > 0:
        payload["price"] = str(price)
    return _nobitex_request("POST", f"/positions/{int(position_id)}/close", payload=payload)


def nobitex_add_order(
    symbol: str,
    side: str,
    price: str,
    amount: str,
    order_type: str = "limit",
):
    try:
        base_cur, dst_cur = _nobitex_symbol_parts(symbol)
        side = str(side).strip().lower()
        order_type = str(order_type).strip().lower()

        if side not in {"buy", "sell"}:
            return False, "جهت سفارش باید buy یا sell باشد."

        if order_type not in {"limit", "market", "stop_market", "stop_limit"}:
            return False, f"نوع اجرای سفارش پشتیبانی نمی‌شود: {order_type}"

        payload = {
            "type": side,
            "srcCurrency": base_cur,
            "dstCurrency": dst_cur,
            "amount": str(amount).strip(),
        }

        # Nobitex requires price for a limit order; market orders don't.
        if order_type != "market":
            payload["price"] = str(price).strip()

        if order_type != "limit":
            payload["execution"] = order_type

        return _nobitex_request(
            "POST",
            "/market/orders/add",
            payload=payload,
        )
    except Exception as e:
        return False, str(e)


def nobitex_get_orders(status: str = "all", symbol: str = ""):
    try:
        params = {}
        status = str(status or "").strip().lower()
        if status and status != "all":
            params["status"] = status

        if symbol:
            base_cur, dst_cur = _nobitex_symbol_parts(symbol)
            params["srcCurrency"] = base_cur
            params["dstCurrency"] = dst_cur

        # Current Nobitex API documents this as GET, not POST JSON.
        return _nobitex_request(
            "GET",
            "/market/orders/list",
            params=params,
        )
    except Exception as e:
        return False, str(e)


def nobitex_cancel_order(order_id: str):
    try:
        payload = {
            "order": int(order_id)
            if str(order_id).strip().isdigit()
            else str(order_id).strip(),
            "status": "canceled",
        }
        # Current docs use "order", not "id".
        return _nobitex_request(
            "POST",
            "/market/orders/update-status",
            payload=payload,
        )
    except Exception as e:
        return False, str(e)


def nobitex_withdraw(
    currency: str,
    address: str,
    amount: str,
    network: str = "",
    otp: str = "",
):
    """Register a crypto withdrawal using the user's wallet id + destination."""
    try:
        cur = str(currency).strip().lower()
        dest = str(address).strip()
        amt = str(amount).strip()
        if not cur or not dest or not amt:
            return False, "ارز، آدرس و مقدار برداشت الزامی است."

        # Current Nobitex API documents wallet (integer id) as the required
        # source wallet field. Resolve it automatically by currency.
        ok_wallets, wallet_result = nobitex_get_wallets()
        if not ok_wallets:
            return False, f"دریافت کیف پول برای برداشت ناموفق: {wallet_result}"

        wallets = wallet_result.get("wallets", []) if isinstance(wallet_result, dict) else []
        wallet_id = None
        for w in wallets:
            if str(w.get("currency", "")).strip().lower() == cur:
                wallet_id = w.get("id")
                break
        if wallet_id is None:
            return False, f"کیف پول {cur} در حساب پیدا نشد."

        payload = {
            "wallet": int(wallet_id),
            "address": dest,
            "amount": amt,
        }
        if network:
            payload["network"] = str(network).strip()

        extra = {}
        if otp:
            extra["X-TOTP"] = str(otp).strip()

        return _nobitex_request(
            "POST",
            "/users/wallets/withdraw",
            payload=payload,
            extra_headers=extra,
        )
    except Exception as e:
        return False, str(e)

def nobitex_deposit_address(currency: str, network: str = ""):
    try:
        payload = {"currency": str(currency).strip().lower()}
        # Current official endpoint is /users/wallets/generate-address.
        if network:
            # The endpoint currently documents currency/wallet; network is not
            # part of the standard request, so don't send an unsupported field.
            logging.info(
                "Deposit address request: network=%s is ignored; "
                "use the wallet's depositInfo for network-specific addresses.",
                network,
            )

        return _nobitex_request(
            "POST",
            "/users/wallets/generate-address",
            payload=payload,
        )
    except Exception as e:
        return False, str(e)


# Load saved credentials before the GUI is initialized.
try:
    _load_nobitex_config()
except Exception:
    pass


# ============================================================
# V5 LOW-PRESSURE ADAPTIVE RESEARCH / PERSISTENT HISTORY ENGINE
# ============================================================
# This layer is deliberately added on top of the existing engine. It does not
# replace the existing SMC/ML/scalper logic; it supplies historical research,
# parameter optimization, support/resistance, volume intelligence, sentiment,
# persistent storage and an empirical ML champion selector.

ADAPTIVE_DB_FILE = str(APP_DATA_DIR / "adaptive_research.sqlite3")
ADAPTIVE_FNG_URL = "https://api.alternative.me/fng/?limit=1&format=json"
ADAPTIVE_HISTORY_YEARS = 2
ADAPTIVE_HISTORY_CHUNK_DAYS = 7
ADAPTIVE_API_MAX_CANDLES = 500
ADAPTIVE_SAFE_CANDLES_PER_REQUEST = 499
ADAPTIVE_BOOTSTRAP_ROWS = 1000
ADAPTIVE_BACKFILL_PAUSE_SEC = 0.35
# Responsiveness safeguards: historical acquisition is deliberately gentle.
UNIFIED_DATA_REQUESTS_PER_UNIT = 2
UNIFIED_BACKFILL_PAGES_PER_UNIT = 1
UNIFIED_DASHBOARD_REFRESH_MS = 8000
UNIFIED_DASHBOARD_REBUILD_MS = 15000
UNIFIED_TABLE_MAX_ROWS = 800
PIPELINE_VERSION = "V7_HIGH_TO_LOW_PHASED_WITH_FULL_AUDIT"
# Process the whole market in timeframe phases: all 1m symbols first,
# then all 5m, 15m, 30m, 1h, 4h and finally 1d. 12h is intentionally
# excluded from the default learning queue to reduce load and match the
# requested training order.
ADAPTIVE_RESEARCH_TIMEFRAMES = [1440, 240, 60, 30, 15, 5, 1]
ADAPTIVE_TIMEFRAME_PHASE_PAUSE_SEC = 2.0
ADAPTIVE_MIN_TRAIN_ROWS = 800
ADAPTIVE_MAX_OPT_ROWS = 25000
# Candidate screening is bounded for runtime, but final OOS validation uses every candle.
ADAPTIVE_FINAL_OOS_FULL_SCAN = True
ADAPTIVE_CACHE_DAYS = 3650
ADAPTIVE_SCHEMA_VERSION = 3

# The grid is intentionally broad enough to include values such as 46/53/70
# and many neighboring values. It is searched in stages so the program does
# not attempt the mathematically impossible Cartesian product of every value.
ADAPTIVE_GRID = {
    "rsi_period": list(range(5, 31)),
    "rsi_os": list(range(20, 46)),
    "rsi_ob": list(range(55, 81)),
    "macd_fast": list(range(5, 21)),
    "macd_slow": list(range(16, 41)),
    "macd_signal": list(range(3, 16)),
    "ema_fast": list(range(5, 31)),
    "ema_slow": list(range(20, 101, 5)),
    "bb_period": list(range(10, 51, 2)),
    "bb_std_x10": list(range(15, 31)),
    "adx_period": list(range(7, 31)),
    "adx_threshold": list(range(15, 36)),
    "stoch_period": list(range(5, 22)),
    "stoch_os": list(range(10, 36)),
    "stoch_ob": list(range(65, 91)),
    "volume_lookback": list(range(10, 101, 5)),
    "volume_factor_x100": list(range(100, 301, 10)),
}

_adaptive_lock = threading.RLock()
_adaptive_initialized = False
_adaptive_last_fng = {"value": None, "classification": "نامشخص", "timestamp": 0.0}
_adaptive_champion_cache = {}
_adaptive_research_status = {"running": False, "message": "آماده", "progress": 0}


def adaptive_db():
    conn = sqlite3.connect(ADAPTIVE_DB_FILE, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def adaptive_init_db():
    global _adaptive_initialized
    with _adaptive_lock:
        if _adaptive_initialized:
            return
        with adaptive_db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY, value TEXT
            );
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL, timeframe INTEGER NOT NULL,
                ts INTEGER NOT NULL, open REAL NOT NULL, high REAL NOT NULL,
                low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
                PRIMARY KEY(symbol,timeframe,ts)
            );
            CREATE INDEX IF NOT EXISTS idx_candles_tf_ts
                ON candles(symbol,timeframe,ts);
            CREATE TABLE IF NOT EXISTS fng (
                ts INTEGER PRIMARY KEY, value REAL, classification TEXT, source TEXT
            );
            CREATE TABLE IF NOT EXISTS optimizer_results (
                symbol TEXT NOT NULL, timeframe INTEGER NOT NULL,
                tested_at INTEGER NOT NULL, method TEXT NOT NULL,
                params_json TEXT NOT NULL, direction TEXT NOT NULL,
                trades INTEGER, wins INTEGER, win_rate REAL,
                profit_factor REAL, expectancy REAL, max_drawdown REAL,
                score REAL, oos INTEGER DEFAULT 1,
                PRIMARY KEY(symbol,timeframe,method,params_json,direction)
            );
            CREATE INDEX IF NOT EXISTS idx_opt_rank
                ON optimizer_results(symbol,timeframe,score DESC);
            CREATE TABLE IF NOT EXISTS model_results (
                symbol TEXT NOT NULL, timeframe INTEGER NOT NULL,
                model TEXT NOT NULL, trained_at INTEGER NOT NULL,
                samples INTEGER, accuracy REAL, auc REAL,
                precision_val REAL, recall_val REAL, f1 REAL,
                brier REAL, artifact TEXT, is_champion INTEGER DEFAULT 0,
                PRIMARY KEY(symbol,timeframe,model)
            );
            CREATE TABLE IF NOT EXISTS research_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, started_at INTEGER,
                finished_at INTEGER, symbol TEXT, timeframe INTEGER,
                status TEXT, message TEXT
            );
            CREATE TABLE IF NOT EXISTS history_audits (
                symbol TEXT NOT NULL, timeframe INTEGER NOT NULL,
                requested_years REAL NOT NULL, expected_candles INTEGER NOT NULL,
                actual_candles INTEGER NOT NULL, missing_candles INTEGER NOT NULL,
                coverage_pct REAL NOT NULL, first_ts INTEGER, last_ts INTEGER,
                gap_count INTEGER NOT NULL DEFAULT 0, largest_gap_candles INTEGER NOT NULL DEFAULT 0,
                complete INTEGER NOT NULL DEFAULT 0, checked_at INTEGER NOT NULL,
                PRIMARY KEY(symbol,timeframe,requested_years)
            );
            CREATE TABLE IF NOT EXISTS learning_units (
                symbol TEXT NOT NULL, timeframe INTEGER NOT NULL,
                requested_years REAL NOT NULL DEFAULT 2,
                status TEXT NOT NULL DEFAULT 'new',
                data_rows INTEGER NOT NULL DEFAULT 0,
                coverage_pct REAL NOT NULL DEFAULT 0,
                last_data_ts INTEGER,
                last_train_ts INTEGER,
                model_count INTEGER NOT NULL DEFAULT 0,
                message TEXT,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY(symbol,timeframe,requested_years)
            );
            CREATE INDEX IF NOT EXISTS idx_learning_units_status ON learning_units(status);
            CREATE TABLE IF NOT EXISTS adaptive_backfill_state (
                symbol TEXT NOT NULL, timeframe INTEGER NOT NULL, cursor_ts INTEGER NOT NULL,
                target_start INTEGER NOT NULL, target_end INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY(symbol,timeframe)
            );
            """)
            db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                       (str(ADAPTIVE_SCHEMA_VERSION),))
        _adaptive_initialized = True


def adaptive_save_candles(symbol, timeframe, df):
    if df is None or df.empty:
        return 0
    adaptive_init_db()
    rows=[]
    for idx,row in df.iterrows():
        try:
            ts=int(pd.Timestamp(idx).timestamp())
            vals=(canonical(symbol), int(timeframe), ts,
                  float(row.open), float(row.high), float(row.low),
                  float(row.close), float(row.volume))
            rows.append(vals)
        except Exception:
            continue
    if not rows:
        return 0
    with adaptive_db() as db:
        db.executemany("""INSERT OR REPLACE INTO candles
            (symbol,timeframe,ts,open,high,low,close,volume)
            VALUES(?,?,?,?,?,?,?,?)""", rows)
    return len(rows)


def adaptive_load_candles(symbol, timeframe, start_ts=None, end_ts=None):
    adaptive_init_db()
    q="SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND timeframe=?"
    args=[canonical(symbol), int(timeframe)]
    if start_ts is not None:
        q += " AND ts>=?"; args.append(int(start_ts))
    if end_ts is not None:
        q += " AND ts<=?"; args.append(int(end_ts))
    q += " ORDER BY ts"
    try:
        with adaptive_db() as db:
            rows=db.execute(q,args).fetchall()
        if not rows:
            return pd.DataFrame()
        d=pd.DataFrame(rows,columns=["ts","open","high","low","close","volume"])
        d.index=pd.to_datetime(d.pop("ts"),unit="s")
        return d.astype(float)
    except Exception:
        return pd.DataFrame()


def _adaptive_completed_window(years, timeframe):
    """Return an exact window containing only completed candles."""
    step=max(60,int(timeframe)*60)
    now=int(time.time())
    end=(now//step)*step-step
    start=end-int(float(years)*365.25*86400)
    start=(start//step)*step
    expected=max(0,((end-start)//step)+1)
    return start,end,step,expected


def adaptive_history_coverage(symbol, timeframe, years=None):
    """Audit exact candle coverage, including gaps, for the requested window."""
    adaptive_init_db()
    key=canonical(symbol); tf=int(timeframe)
    if years is None:
        with adaptive_db() as db:
            row=db.execute("SELECT MAX(requested_years) FROM history_audits WHERE symbol=? AND timeframe=?",(key,tf)).fetchone()
        years=float(row[0]) if row and row[0] else float(ADAPTIVE_HISTORY_YEARS)
    start,end,step,expected=_adaptive_completed_window(float(years),tf)
    try:
        with adaptive_db() as db:
            rows=db.execute("SELECT ts FROM candles WHERE symbol=? AND timeframe=? AND ts>=? AND ts<=? ORDER BY ts",(key,tf,start,end)).fetchall()
        ts=[int(r[0]) for r in rows]
        actual=len(ts)
        missing=max(0,expected-actual)
        gaps=[]
        if ts:
            for a,b in zip(ts,ts[1:]):
                delta=b-a
                if delta>step:
                    gaps.append(max(1,int(round(delta/step))-1))
        gap_count=len(gaps)
        largest_gap=max(gaps) if gaps else 0
        first=ts[0] if ts else None; last=ts[-1] if ts else None
        endpoint_ok=(first==start and last==end)
        complete=(expected>0 and actual==expected and gap_count==0 and endpoint_ok)
        coverage=(actual/expected*100.0) if expected else 0.0
        # A unit becomes trainable as soon as it has enough chronological data.
        # Full requested coverage remains a separate state and is filled in the
        # background. This prevents a multi-year 1m backfill from blocking ML.
        trainable = bool(actual >= ADAPTIVE_MIN_TRAIN_ROWS and last is not None)
        recent_ok = bool(last is not None and last >= (end - (3 * step)))
        return {"symbol":key,"timeframe":tf,"years":float(years),"start":start,"end":end,"step":step,
                "expected":expected,"count":actual,"missing":missing,"coverage_pct":coverage,
                "first":first,"last":last,"gap_count":gap_count,"largest_gap_candles":largest_gap,
                "complete":bool(complete),"trainable":trainable,"recent_ok":recent_ok}
    except Exception as exc:
        return {"symbol":key,"timeframe":tf,"years":float(years),"start":start,"end":end,"step":step,
                "expected":expected,"count":0,"missing":expected,"coverage_pct":0.0,
                "first":None,"last":None,"gap_count":0,"largest_gap_candles":0,
                "complete":False,"trainable":False,"recent_ok":False,"error":str(exc)}


def adaptive_save_history_audit(audit):
    try:
        adaptive_init_db()
        with adaptive_db() as db:
            db.execute("""INSERT OR REPLACE INTO history_audits
                (symbol,timeframe,requested_years,expected_candles,actual_candles,missing_candles,coverage_pct,
                 first_ts,last_ts,gap_count,largest_gap_candles,complete,checked_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (audit["symbol"],audit["timeframe"],audit["years"],audit["expected"],audit["count"],audit["missing"],
                 audit["coverage_pct"],audit.get("first"),audit.get("last"),audit["gap_count"],
                 audit["largest_gap_candles"],1 if audit["complete"] else 0,int(time.time())))
    except Exception as exc:
        logging.debug("history audit save failed: %s",exc)


def adaptive_fetch_and_store_history(symbol, timeframe, years=2, force=False, max_requests=None):
    """Fetch missing candles using API-safe <=500-candle pages.

    The old implementation requested seven-day chunks even for 1m candles.
    Nobitex history requests are capped at 500 candles, so those chunks could
    never become complete. This implementation paginates by candle count,
    starts from the newest completed candles so training can begin quickly,
    and optionally continues backwards for a bounded number of requests.
    """
    adaptive_init_db()
    key=canonical(symbol); tf=int(timeframe)
    start_ts,end_ts,step,expected=_adaptive_completed_window(float(years),tf)
    page_candles=max(10,min(ADAPTIVE_API_MAX_CANDLES,ADAPTIVE_SAFE_CANDLES_PER_REQUEST))
    page_span=(page_candles-1)*step
    requests_done=0

    def fetch_range(a,b):
        nonlocal requests_done
        try:
            j=robust_history_udf(key,tf,int(a),int(b),valid_symbols=set(valid_symbols_map.keys()) if valid_symbols_map else None)
            d=_deep_df_from_udf(j)
            if d is not None and not d.empty:
                adaptive_save_candles(key,tf,d)
                requests_done += 1
                return len(d)
        except Exception as exc:
            logging.debug("adaptive history page failed %s tf=%s %s-%s: %s",key,tf,a,b,exc)
        requests_done += 1
        return 0

    # Always prioritize the newest data. This makes the unit trainable quickly.
    bootstrap_end=end_ts
    bootstrap_start=max(start_ts, end_ts-(max(ADAPTIVE_BOOTSTRAP_ROWS,ADAPTIVE_MIN_TRAIN_ROWS*2)-1)*step)
    cur=bootstrap_start
    while cur<=bootstrap_end and (max_requests is None or requests_done<max_requests):
        to=min(bootstrap_end,cur+page_span)
        if force:
            fetch_range(cur,to)
        else:
            with adaptive_db() as db:
                row=db.execute("SELECT COUNT(*),MIN(ts),MAX(ts) FROM candles WHERE symbol=? AND timeframe=? AND ts>=? AND ts<=?",
                               (key,tf,cur,to)).fetchone()
            expected_page=max(1,((to-cur)//step)+1)
            if not row or int(row[0] or 0)<expected_page or int(row[1] or 0)!=cur or int(row[2] or 0)!=to:
                fetch_range(cur,to)
        cur=to+step

    # If explicitly requested without a request budget, continue backwards and
    # complete the full requested window. Research uses a bounded bootstrap and
    # the background scheduler calls this function repeatedly for the rest.
    if max_requests is None:
        cur=start_ts
        while cur<bootstrap_start:
            to=min(bootstrap_start-step,cur+page_span)
            with adaptive_db() as db:
                row=db.execute("SELECT COUNT(*),MIN(ts),MAX(ts) FROM candles WHERE symbol=? AND timeframe=? AND ts>=? AND ts<=?",
                               (key,tf,cur,to)).fetchone()
            expected_page=max(1,((to-cur)//step)+1)
            if force or not row or int(row[0] or 0)<expected_page or int(row[1] or 0)!=cur or int(row[2] or 0)!=to:
                fetch_range(cur,to)
            cur=to+step

    audit=adaptive_history_coverage(key,tf,years)
    adaptive_save_history_audit(audit)
    return adaptive_load_candles(key,tf,start_ts,end_ts)


def _adaptive_set_unit_status(symbol,timeframe,years,status,message=None,audit=None,model_count=None):
    try:
        adaptive_init_db(); audit=audit or adaptive_history_coverage(symbol,timeframe,years)
        now=int(time.time())
        if model_count is None:
            with adaptive_db() as db:
                r=db.execute("SELECT COUNT(*) FROM model_results WHERE symbol=? AND timeframe=? AND is_champion=1",(canonical(symbol),int(timeframe))).fetchone()
                model_count=int(r[0] or 0)
        with adaptive_db() as db:
            db.execute("""INSERT OR REPLACE INTO learning_units
                (symbol,timeframe,requested_years,status,data_rows,coverage_pct,last_data_ts,last_train_ts,model_count,message,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (canonical(symbol),int(timeframe),float(years),str(status),int(audit.get('count',0)),float(audit.get('coverage_pct',0)),
                 audit.get('last'),None,model_count,str(message or ''),now))
    except Exception as exc:
        logging.debug("learning unit status failed: %s",exc)


def adaptive_backfill_unit(symbol,timeframe,years=2,chunk_requests=6):
    """Resume old-history backfill from a persistent cursor.

    Unlike the previous scanner, this does not rescan thousands of 1m pages on
    every pass. Each unit remembers its next page in SQLite and advances it
    after every successful/complete page. This makes multi-day backfill truly
    resumable.
    """
    adaptive_init_db(); key=canonical(symbol); tf=int(timeframe)
    start_ts,end_ts,step,_=_adaptive_completed_window(float(years),tf)
    page_span=(ADAPTIVE_SAFE_CANDLES_PER_REQUEST-1)*step
    bootstrap_start=max(start_ts, end_ts-(max(ADAPTIVE_BOOTSTRAP_ROWS,ADAPTIVE_MIN_TRAIN_ROWS*2)-1)*step)
    with adaptive_db() as db:
        row=db.execute("SELECT cursor_ts,target_start,target_end FROM adaptive_backfill_state WHERE symbol=? AND timeframe=?",(key,tf)).fetchone()
        if row is None or int(row[1])!=start_ts or int(row[2])!=end_ts:
            cursor=start_ts
            db.execute("INSERT OR REPLACE INTO adaptive_backfill_state(symbol,timeframe,cursor_ts,target_start,target_end,updated_at) VALUES(?,?,?,?,?,?)",(key,tf,cursor,start_ts,end_ts,int(time.time())))
        else:
            cursor=int(row[0])
    done_pages=0
    while cursor < bootstrap_start and done_pages < max(1,int(chunk_requests)):
        to=min(bootstrap_start-step,cursor+page_span)
        expected_page=max(1,((to-cursor)//step)+1)
        with adaptive_db() as db:
            r=db.execute("SELECT COUNT(*),MIN(ts),MAX(ts) FROM candles WHERE symbol=? AND timeframe=? AND ts>=? AND ts<=?",(key,tf,cursor,to)).fetchone()
        complete_page=bool(r and int(r[0] or 0)>=expected_page and int(r[1] or 0)==cursor and int(r[2] or 0)==to)
        if not complete_page:
            try:
                time.sleep(float(ADAPTIVE_BACKFILL_PAUSE_SEC))
                j=robust_history_udf(key,tf,int(cursor),int(to),valid_symbols=set(valid_symbols_map.keys()) if valid_symbols_map else None)
                d=_deep_df_from_udf(j)
                if d is not None and not d.empty:
                    adaptive_save_candles(key,tf,d)
            except Exception as exc:
                logging.debug("backfill page failed %s %s: %s",key,tf,exc)
                # Do not advance the cursor on failure; next run retries the same page.
                break
        cursor=to+step
        done_pages+=1
        with adaptive_db() as db:
            db.execute("UPDATE adaptive_backfill_state SET cursor_ts=?,updated_at=? WHERE symbol=? AND timeframe=?",(cursor,int(time.time()),key,tf))
    audit=adaptive_history_coverage(key,tf,years); adaptive_save_history_audit(audit)
    return audit

def adaptive_start_background_backfill(symbols,timeframes,years=2):
    """Continuously complete historical gaps with a low-rate resumable worker."""
    def worker():
        # IMPORTANT: timeframe-major order. Finish the market-wide pass for
        # one timeframe before moving to the next timeframe. This prevents a
        # single symbol from monopolizing the downloader and makes progress
        # visible as: 1m(all symbols) -> 5m(all) -> 15m(all) -> ... -> 1d(all).
        units=[(s,int(t)) for t in list(timeframes) for s in list(symbols)]
        idle_rounds=0
        while units and idle_rounds < 30:
            if globals().get('_adaptive_backfill_stop',False): break
            progressed=False; all_complete=True
            for idx,(sym,tf) in enumerate(units):
                if globals().get('_adaptive_backfill_stop',False): break
                try:
                    audit=adaptive_backfill_unit(sym,tf,years=years,chunk_requests=UNIFIED_BACKFILL_PAGES_PER_UNIT)
                    if audit.get('complete'):
                        _adaptive_set_unit_status(sym,tf,years,'ready','تاریخچه کامل و مدل قابل استفاده',audit)
                        if '_unified_upsert_state' in globals():
                            _unified_upsert_state(sym,tf,history_years=float(years),history_last=int(audit.get('last') or 0),history_count=int(audit.get('count',0)),history_expected=int(audit.get('expected',0)),coverage=float(audit.get('coverage_pct',0)),status='ready',message='تاریخچه کامل')
                    elif audit.get('trainable'):
                        _adaptive_set_unit_status(sym,tf,years,'ready_partial','آموزش آماده؛ تکمیل تاریخچه در پس‌زمینه',audit)
                        if '_unified_upsert_state' in globals():
                            _unified_upsert_state(sym,tf,history_years=float(years),history_last=int(audit.get('last') or 0),history_count=int(audit.get('count',0)),history_expected=int(audit.get('expected',0)),coverage=float(audit.get('coverage_pct',0)),status='ready_partial',message='آموزش آماده؛ تکمیل تاریخچه ادامه دارد')
                        all_complete=False; progressed=True
                    else:
                        _adaptive_set_unit_status(sym,tf,years,'needs_data','در حال جمع‌آوری تاریخچه',audit)
                        if '_unified_upsert_state' in globals():
                            _unified_upsert_state(sym,tf,history_years=float(years),history_last=int(audit.get('last') or 0),history_count=int(audit.get('count',0)),history_expected=int(audit.get('expected',0)),coverage=float(audit.get('coverage_pct',0)),status='needs_data',message='در حال جمع‌آوری تاریخچه')
                        all_complete=False; progressed=True
                    if audit.get('complete'): progressed=True
                except Exception as exc:
                    all_complete=False; logging.debug("background backfill %s %s: %s",sym,tf,exc)
            if all_complete: break
            idle_rounds=0 if progressed else idle_rounds+1
            time.sleep(1.5 if progressed else 8.0)
    global _adaptive_backfill_thread, _adaptive_backfill_stop
    try: _adaptive_backfill_stop=False
    except Exception: pass
    old=globals().get('_adaptive_backfill_thread')
    if old is not None and old.is_alive(): return old
    t=threading.Thread(target=worker,daemon=True,name="AdaptiveHistoryBackfill")
    _adaptive_backfill_thread=t; t.start(); return t


def adaptive_fear_greed(force=False):
    global _adaptive_last_fng
    adaptive_init_db()
    now=time.time()
    if not force and _adaptive_last_fng.get("value") is not None and now-_adaptive_last_fng["timestamp"] < 6*3600:
        return dict(_adaptive_last_fng)
    try:
        r=get_session().get(ADAPTIVE_FNG_URL,timeout=(2.5,4.0))
        if r.status_code != 200:
            raise RuntimeError(f"fng-http-{r.status_code}")
        j=r.json()
        item=(j.get("data") or [{}])[0]
        value=float(item.get("value"))
        cls=str(item.get("value_classification") or "نامشخص")
        ts=int(item.get("timestamp") or now)
        with adaptive_db() as db:
            db.execute("INSERT OR REPLACE INTO fng(ts,value,classification,source) VALUES(?,?,?,?)",
                       (ts,value,cls,"alternative.me"))
        _adaptive_last_fng={"value":value,"classification":cls,"timestamp":now}
        return dict(_adaptive_last_fng)
    except Exception:
        try:
            with adaptive_db() as db:
                row=db.execute("SELECT value,classification,ts FROM fng ORDER BY ts DESC LIMIT 1").fetchone()
            if row:
                _adaptive_last_fng={"value":float(row[0]),"classification":row[1],"timestamp":now}
                return dict(_adaptive_last_fng)
        except Exception:
            pass
    return {"value":None,"classification":"نامشخص","timestamp":0.0}


def adaptive_support_resistance(df):
    """Return a consensus of structural, equal, pivot and volume-profile levels."""
    out={"supports":[],"resistances":[],"support":None,"resistance":None,
         "support_score":0.0,"resistance_score":0.0}
    try:
        if df is None or len(df)<30:
            return out
        price=float(df.close.iloc[-1])
        candidates_s=[]; candidates_r=[]
        sh,sl=detect_swing_points(df,left=3,right=3)
        for i in sl[-20:]: candidates_s.append((float(df.low.iloc[i]),2.0,"Swing"))
        for i in sh[-20:]: candidates_r.append((float(df.high.iloc[i]),2.0,"Swing"))
        eq=detect_equal_high_low(df,tolerance_pct=.20)
        for x in eq.get("eql",[]): candidates_s.append((float(x),2.5,"EqualLow"))
        for x in eq.get("eqh",[]): candidates_r.append((float(x),2.5,"EqualHigh"))
        try:
            poc,hvn,lvn=calculate_volume_profile(df)
            if poc:
                (candidates_s if poc<=price else candidates_r).append((float(poc),3.0,"POC"))
            for x in hvn[-8:]:
                (candidates_s if x<=price else candidates_r).append((float(x),2.0,"HVN"))
        except Exception:
            pass
        # Classic daily-style pivots from the available last candle.
        h=float(df.high.iloc[-1]); l=float(df.low.iloc[-1]); c=float(df.close.iloc[-1])
        pp=(h+l+c)/3.0; r1=2*pp-l; s1=2*pp-h; r2=pp+(h-l); s2=pp-(h-l)
        for x in (s1,s2):
            if x>0 and x<=price: candidates_s.append((x,1.5,"Pivot"))
        for x in (r1,r2):
            if x>price: candidates_r.append((x,1.5,"Pivot"))
        def merge(items):
            groups=[]
            for val,w,src in sorted(items,key=lambda z:z[0]):
                if val<=0: continue
                placed=False
                for g in groups:
                    if abs(val-g[0])/max(g[0],1e-12)*100 <= 0.25:
                        g[0]=(g[0]*g[1]+val*w)/(g[1]+w); g[1]+=w; g[2].append(src); placed=True; break
                if not placed: groups.append([val,w,[src]])
            return [(float(g[0]),float(g[1]),"+".join(sorted(set(g[2])))) for g in groups]
        candidates_s=[x for x in candidates_s if x[0] <= price]
        candidates_r=[x for x in candidates_r if x[0] >= price]
        ss=merge(candidates_s); rr=merge(candidates_r)
        ss=sorted(ss,key=lambda x: x[0],reverse=True)
        rr=sorted(rr,key=lambda x: x[0])
        out["supports"]=[x[0] for x in ss[:8]]; out["resistances"]=[x[0] for x in rr[:8]]
        if ss: out["support"]=ss[0][0]; out["support_score"]=min(100,ss[0][1]*20)
        if rr: out["resistance"]=rr[0][0]; out["resistance_score"]=min(100,rr[0][1]*20)
        out["support_sources"]=[x[2] for x in ss[:8]]; out["resistance_sources"]=[x[2] for x in rr[:8]]
    except Exception:
        pass
    return out


def adaptive_volume_features(df):
    out={"ratio":1.0,"z":0.0,"obv_slope":0.0,"mfi":50.0,"cmf":0.0,
         "vwap_distance":0.0,"poc_distance":0.0,"pressure":0.0,"state":"عادی"}
    try:
        if df is None or len(df)<30: return out
        v=pd.to_numeric(df.volume,errors="coerce").fillna(0)
        c=pd.to_numeric(df.close,errors="coerce")
        h=pd.to_numeric(df.high,errors="coerce"); l=pd.to_numeric(df.low,errors="coerce")
        o=pd.to_numeric(df.open,errors="coerce")
        recent=float(v.tail(6).mean()); base=float(v.tail(48).mean()); out["ratio"]=recent/base if base>0 else 1.0
        mu=float(v.tail(100).mean()); sd=float(v.tail(100).std()); out["z"]=(float(v.iloc[-1])-mu)/(sd+1e-12)
        obv=np.where(c.diff().fillna(0)>=0,v,-v); obv_s=pd.Series(obv,index=df.index).cumsum()
        if len(obv_s)>=20: out["obv_slope"]=float((obv_s.iloc[-1]-obv_s.iloc[-20])/(abs(obv_s.iloc[-20])+1e-12))
        try: out["mfi"]=float(calculate_mfi(df))
        except Exception: pass
        mf=((c-l)-(h-c))/((h-l).replace(0,np.nan)); mf=mf.fillna(0)
        vol20=float(v.tail(20).sum()); out["cmf"]=float((mf.tail(20)*v.tail(20)).sum()/(vol20+1e-12))
        tp=(h+l+c)/3; vs=float(v.tail(24).sum())
        if vs>0:
            vw=float((tp*v).tail(24).sum()/vs); out["vwap_distance"]=float((c.iloc[-1]-vw)/(vw+1e-12)*100)
        try:
            poc,_,_=calculate_volume_profile(df)
            if poc: out["poc_distance"]=float((c.iloc[-1]-poc)/(poc+1e-12)*100)
        except Exception: pass
        close_pos=((c-o)/(h-l).replace(0,np.nan)).fillna(0).tail(10)
        out["pressure"]=float(np.clip((close_pos*v.tail(10)).sum()/(v.tail(10).sum()+1e-12),-1,1))
        if out["ratio"]>=3 or out["z"]>=3: out["state"]="انفجاری"
        elif out["ratio"]>=1.5 or out["z"]>=1.5: out["state"]="بالا"
        elif out["ratio"]<=0.7: out["state"]="کم"
    except Exception:
        pass
    return out


def adaptive_indicator_signal(df,p):
    """Parameterized RSI+MACD+EMA+BB+ADX+Stoch signal used by optimizer."""
    try:
        if df is None or len(df)<max(80,int(p["ema_slow"])+5): return 0
        c=df.close.astype(float)
        rsi_s=compute_rsi_series(df,int(p["rsi_period"]))
        r=float(rsi_s.iloc[-1]) if np.isfinite(rsi_s.iloc[-1]) else 50.0
        fast=int(p["macd_fast"]); slow=int(p["macd_slow"]); sig=int(p["macd_signal"])
        emaf=c.ewm(span=fast,adjust=False).mean(); emas=c.ewm(span=slow,adjust=False).mean()
        macd_line=emaf-emas; signal_line=macd_line.ewm(span=sig,adjust=False).mean(); hist=float(macd_line.iloc[-1]-signal_line.iloc[-1])
        ema_fast=c.ewm(span=int(p["ema_fast"]),adjust=False).mean().iloc[-1]
        ema_slow=c.ewm(span=int(p["ema_slow"]),adjust=False).mean().iloc[-1]
        bb_p=int(p["bb_period"]); bb_std=float(p["bb_std_x10"])/10.0
        mid=c.rolling(bb_p).mean().iloc[-1]; sd=c.rolling(bb_p).std().iloc[-1]
        ub=mid+bb_std*sd; lb=mid-bb_std*sd
        adx=float(calculate_adx(df,int(p["adx_period"])))
        k=pd.Series((df.high-df.low).rolling(int(p["stoch_period"])).sum())
        # robust stochastic implementation
        lo=df.low.rolling(int(p["stoch_period"])).min().iloc[-1]; hi=df.high.rolling(int(p["stoch_period"])).max().iloc[-1]
        stoch=(float(c.iloc[-1])-lo)/(hi-lo+1e-12)*100
        score=0
        if r<=p["rsi_os"]: score+=2
        elif r>=p["rsi_ob"]: score-=2
        if hist>0: score+=2
        elif hist<0: score-=2
        if ema_fast>ema_slow: score+=1
        elif ema_fast<ema_slow: score-=1
        if float(c.iloc[-1])<=lb: score+=1
        elif float(c.iloc[-1])>=ub: score-=1
        if adx>=p["adx_threshold"]:
            score += 1 if hist>0 else -1 if hist<0 else 0
        if stoch<=p["stoch_os"]: score+=1
        elif stoch>=p["stoch_ob"]: score-=1
        return 1 if score>=3 else -1 if score<=-3 else 0
    except Exception:
        return 0


def adaptive_trade_outcome(df, i, side, horizon=12, tp_pct=1.0, sl_pct=0.6):
    try:
        entry=float(df.close.iloc[i]); future=df.iloc[i+1:i+1+horizon]
        if future.empty or entry<=0: return None
        if side==1:
            tp=entry*(1+tp_pct/100); sl=entry*(1-sl_pct/100)
            for _,r in future.iterrows():
                hit_tp=float(r.high)>=tp; hit_sl=float(r.low)<=sl
                if hit_tp and hit_sl: return -1  # conservative same-candle assumption
                if hit_tp: return 1
                if hit_sl: return -1
            return 0
        else:
            tp=entry*(1-tp_pct/100); sl=entry*(1+sl_pct/100)
            for _,r in future.iterrows():
                hit_tp=float(r.low)<=tp; hit_sl=float(r.high)>=sl
                if hit_tp and hit_sl: return -1
                if hit_tp: return 1
                if hit_sl: return -1
            return 0
    except Exception:
        return None


def adaptive_backtest_params(df, params, direction, max_rows=ADAPTIVE_MAX_OPT_ROWS):
    if df is None or len(df)<200: return None
    n=len(df); start=max(80,int(params["ema_slow"])+10); end=n-13
    if end<=start: return None
    # Sample evenly if the dataset is very large. The stored history remains complete.
    scan_n=(end-start) if (max_rows is None or int(max_rows)<=0) else min(int(max_rows),end-start)
    idx=np.arange(start,end,dtype=int) if scan_n >= (end-start) else np.linspace(start,end-1,scan_n,dtype=int)
    wins=losses=flats=0; gross_win=0.0; gross_loss=0.0; returns=[]
    for i in idx:
        sig=adaptive_indicator_signal(df.iloc[:i+1],params)
        if sig!=direction: continue
        out=adaptive_trade_outcome(df,i,direction)
        if out is None: continue
        if out>0: wins+=1; gross_win+=1.0; returns.append(1.0)
        elif out<0: losses+=1; gross_loss+=0.6; returns.append(-0.6)
        else: flats+=1; returns.append(0.0)
    trades=wins+losses+flats
    if trades<20: return None
    wr=100*wins/max(1,wins+losses)
    pf=gross_win/max(gross_loss,1e-12) if gross_win>0 else 0.0
    expectancy=(gross_win-gross_loss)/trades
    # Simple equity drawdown on 1R/0.6R returns.
    eq=1.0; peak=1.0; dd=0.0
    for r in returns:
        eq*=1+r/100.0
        peak=max(peak,eq); dd=max(dd,(peak-eq)/peak*100)
    score=(wr*0.45)+(min(3,pf)/3*100*0.30)+(max(-1,min(1,expectancy))*50*0.15)-min(30,dd)*0.10
    return {"trades":trades,"wins":wins,"win_rate":wr,"profit_factor":pf,
            "expectancy":expectancy,"max_drawdown":dd,"score":score}


def adaptive_param_candidates():
    # Stage 1: single indicators and sensible pair combinations.
    out=[]
    for rsi_p in ADAPTIVE_GRID["rsi_period"]:
        for os in ADAPTIVE_GRID["rsi_os"]:
            for ob in (46,50,53,55,60,65,70,75,80):
                if os>=ob: continue
                out.append({"rsi_period":rsi_p,"rsi_os":os,"rsi_ob":ob,
                            "macd_fast":12,"macd_slow":26,"macd_signal":9,
                            "ema_fast":9,"ema_slow":50,"bb_period":20,"bb_std_x10":20,
                            "adx_period":14,"adx_threshold":20,"stoch_period":14,
                            "stoch_os":20,"stoch_ob":80,"volume_lookback":48,"volume_factor_x100":150})
    # Add representative MACD/EMA/BB/ADX/Stoch candidates. These are later
    # refined around the best region rather than doing an astronomical product.
    base=out[:]
    for mf in (5,8,12,15,20):
        for ms in (20,26,30,35,40):
            if mf>=ms: continue
            for sg in (5,7,9,12,15):
                for rsi_p in (7,10,14,21,28):
                    p=base[0].copy(); p.update(macd_fast=mf,macd_slow=ms,macd_signal=sg,rsi_period=rsi_p)
                    out.append(p)
    for ef in (5,9,12,20,21,30):
        for es in (34,50,55,100):
            if ef>=es: continue
            p=base[0].copy(); p.update(ema_fast=ef,ema_slow=es); out.append(p)
    for bp in (10,14,20,24,30,40,50):
        for bs in (15,18,20,22,25,30):
            p=base[0].copy(); p.update(bb_period=bp,bb_std_x10=bs); out.append(p)
    for ap in (7,10,14,20,21,28,30):
        for at in (15,18,20,22,25,30,35):
            p=base[0].copy(); p.update(adx_period=ap,adx_threshold=at); out.append(p)
    # De-duplicate.
    seen=set(); result=[]
    for p in out:
        k=json.dumps(p,sort_keys=True)
        if k not in seen: seen.add(k); result.append(p)
    return result


def adaptive_refine_params(df, seeds, direction):
    results=[]
    for seed in seeds[:40]:
        candidates=[]
        # Local neighborhood includes every RSI threshold around the winner.
        for os in range(max(20,int(seed["rsi_os"])-5),min(45,int(seed["rsi_os"])+5)+1):
            for ob in range(max(55,int(seed["rsi_ob"])-5),min(80,int(seed["rsi_ob"])+5)+1):
                if os>=ob: continue
                p=seed.copy(); p.update(rsi_os=os,rsi_ob=ob); candidates.append(p)
        for rp in range(max(5,int(seed["rsi_period"])-3),min(30,int(seed["rsi_period"])+3)+1):
            p=seed.copy(); p["rsi_period"]=rp; candidates.append(p)
        for mf in range(max(5,int(seed["macd_fast"])-3),min(20,int(seed["macd_fast"])+3)+1):
            for sg in range(max(3,int(seed["macd_signal"])-2),min(15,int(seed["macd_signal"])+2)+1):
                if mf>=int(seed["macd_slow"]): continue
                p=seed.copy(); p.update(macd_fast=mf,macd_signal=sg); candidates.append(p)
        for p in candidates:
            m=adaptive_backtest_params(df,p,direction)
            if m:
                results.append((m["score"],p,m))
    results.sort(key=lambda x:x[0],reverse=True)
    return results


def adaptive_optimize_symbol_tf(symbol,timeframe,years=2,force_history=False):
    adaptive_init_db()
    key=canonical(symbol); tf=int(timeframe)
    with _adaptive_lock:
        _adaptive_research_status.update(running=True,message=f"داده {key} / {tf}m",progress=0)
    df=adaptive_fetch_and_store_history(key,tf,years=years,force=force_history,max_requests=8)
    audit=adaptive_history_coverage(key,tf,years)
    if not audit.get("trainable"):
        _adaptive_set_unit_status(key,tf,years,'needs_data',f"داده کافی برای آموزش نیست: {audit.get('count',0)}/{ADAPTIVE_MIN_TRAIN_ROWS}",audit)
        with _adaptive_lock: _adaptive_research_status.update(running=False,message=f"داده کافی نیست: {key} {tf}m | {audit.get('count',0)}/{ADAPTIVE_MIN_TRAIN_ROWS}",progress=0)
        return []
    # Use all currently available chronological candles. Full requested coverage
    # is allowed to continue in the background and no longer blocks training.
    df=adaptive_load_candles(key,tf,audit["start"],audit["end"])
    _adaptive_set_unit_status(key,tf,years,'training',f"آماده آموزش: {len(df)} کندل | پوشش {audit.get('coverage_pct',0):.2f}%",audit)
    if df is None or len(df)<ADAPTIVE_MIN_TRAIN_ROWS:
        with _adaptive_lock: _adaptive_research_status.update(running=False,message=f"داده کافی نیست: {key} {tf}m",progress=0)
        return []
    # chronological OOS split: only the training half chooses parameters; final
    # quarter is a genuine unseen validation check.
    cut=int(len(df)*0.70); train=df.iloc[:cut].copy(); test=df.iloc[cut:].copy()
    candidates=adaptive_param_candidates()
    stage=[]
    # Deterministic thinning prevents a 26^2*... explosion while preserving
    # exhaustive coverage of each individual parameter and representative pairs.
    stride=max(1,len(candidates)//1800)
    candidates=candidates[::stride]
    for j,p in enumerate(candidates):
        for direction in (1,-1):
            m=adaptive_backtest_params(train,p,direction)
            if m:
                stage.append((m["score"],p,direction,m))
        if j%50==0:
            with _adaptive_lock: _adaptive_research_status["progress"]=int(j/max(1,len(candidates))*60)
    stage.sort(key=lambda x:x[0],reverse=True)
    winners=[]
    for direction in (1,-1):
        seeds=[x[1] for x in stage if x[2]==direction][:40]
        refined=adaptive_refine_params(train,seeds,direction)
        for score,p,m in refined[:10]:
            # Final OOS check scans every candle in the selected test window.
            oos=adaptive_backtest_params(test,p,direction,max_rows=0 if ADAPTIVE_FINAL_OOS_FULL_SCAN else 10000)
            if oos:
                final_score=0.65*score+0.35*oos["score"]
                winners.append((final_score,p,direction,m,oos))
    winners.sort(key=lambda x:x[0],reverse=True)
    now=int(time.time())
    with adaptive_db() as db:
        db.execute("DELETE FROM optimizer_results WHERE symbol=? AND timeframe=?",(key,tf))
        for final_score,p,direction,m,oos in winners[:30]:
            db.execute("""INSERT OR REPLACE INTO optimizer_results
                (symbol,timeframe,tested_at,method,params_json,direction,trades,wins,win_rate,profit_factor,expectancy,max_drawdown,score,oos)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(key,tf,now,"adaptive_rsi_macd_ensemble",
                json.dumps(p,sort_keys=True),"LONG" if direction==1 else "SHORT",
                m["trades"],m["wins"],m["win_rate"],m["profit_factor"],m["expectancy"],m["max_drawdown"],final_score,1))
    champion=winners[0] if winners else None
    if champion:
        _adaptive_champion_cache[f"{key}|{tf}"]={"params":champion[1],"direction":champion[2],"score":champion[0],"train":champion[3],"oos":champion[4],"tested_at":now}
    with _adaptive_lock: _adaptive_research_status.update(running=False,message=f"پایان {key} / {tf}m",progress=100)
    return winners[:30]


def adaptive_get_champion(symbol,timeframe):
    key=f"{canonical(symbol)}|{int(timeframe)}"
    if key in _adaptive_champion_cache: return _adaptive_champion_cache[key]
    adaptive_init_db()
    try:
        with adaptive_db() as db:
            row=db.execute("""SELECT params_json,direction,score,tested_at,trades,wins,win_rate,profit_factor,max_drawdown
                FROM optimizer_results WHERE symbol=? AND timeframe=? ORDER BY score DESC LIMIT 1""",
                           (canonical(symbol),int(timeframe))).fetchone()
        if row:
            x={"params":json.loads(row[0]),"direction":1 if row[1]=="LONG" else -1,
               "score":float(row[2]),"tested_at":int(row[3]),
               "train":{"trades":int(row[4] or 0),"wins":int(row[5] or 0),"win_rate":float(row[6] or 0),"profit_factor":float(row[7] or 0),"max_drawdown":float(row[8] or 0)}}
            _adaptive_champion_cache[key]=x; return x
    except Exception: pass
    return None


def adaptive_apply_champion(df,symbol,timeframe):
    ch=adaptive_get_champion(symbol,timeframe)
    if not ch or df is None or df.empty: return {"score":50.0,"signal":0,"optimized":False,"champion":None}
    sig=adaptive_indicator_signal(df,ch["params"])
    # Confidence is empirical, not a promise of future profit.
    wr=float(ch.get("train",{}).get("win_rate",50)); score=float(ch.get("score",50))
    confidence=float(np.clip(50+(wr-50)*0.55+(score-50)*0.15,0,99))
    return {"score":score,"signal":sig,"optimized":True,"confidence":confidence,"champion":ch}


def adaptive_ml_features(df):
    try:
        if df is None or len(df)<80: return None
        d=df.copy(); c=d.close.astype(float); v=d.volume.astype(float)
        f=pd.DataFrame(index=d.index)
        f["ret1"]=c.pct_change(1)*100; f["ret3"]=c.pct_change(3)*100; f["ret12"]=c.pct_change(12)*100
        f["rsi"]=compute_rsi_series(d,14); f["rsi7"]=compute_rsi_series(d,7); f["macd"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
        f["atr_pct"]=calculate_atr_series(d,14)/c*100 if 'calculate_atr_series' in globals() else d["high"].rolling(14).max().sub(d["low"].rolling(14).min()).div(c)*100
        f["adx"]=pd.Series([calculate_adx(d.iloc[:i+1],14) if i>=30 else np.nan for i in range(len(d))],index=d.index)
        f["vol_ratio"]=v/(v.rolling(48).mean()+1e-12); f["vol_z"]=(v-v.rolling(100).mean())/(v.rolling(100).std()+1e-12)
        f["ema9_dist"]=(c-c.ewm(span=9,adjust=False).mean())/c*100; f["ema50_dist"]=(c-c.ewm(span=50,adjust=False).mean())/c*100
        f["range_pct"]=(d.high-d.low)/c*100
        f["body_pct"]=(d.close-d.open)/c*100
        return f.replace([np.inf,-np.inf],np.nan).dropna()
    except Exception:
        return None


def adaptive_train_ml_champions(symbol,timeframe,df=None):
    """Train and select ML champions with expanding-window walk-forward OOS.

    No future samples are used to choose the model. Each fold trains only on
    observations before the test window. OOS probabilities from all folds are
    then used to fit an isotonic probability calibrator. The final champion is
    refit on all available historical data and saved together with the
    calibrator and walk-forward metrics.
    """
    if not _ML_AVAILABLE.get("sklearn"): return None
    key=canonical(symbol); tf=int(timeframe)
    if df is None or df.empty: df=adaptive_load_candles(key,tf)
    if df is None or len(df)<ADAPTIVE_MIN_TRAIN_ROWS: return None
    f=adaptive_ml_features(df)
    if f is None or len(f)<500: return None
    horizon=max(2,int(round(60/tf)))
    future=df.close.shift(-horizon)
    y=(future/df.close-1>0).astype(int).reindex(f.index).dropna()
    common=f.index.intersection(y.index)
    X=f.loc[common].copy(); y=y.loc[common].astype(int)
    if len(X)<500 or y.nunique()<2: return None

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,brier_score_loss
    from sklearn.base import clone

    def make_models():
        models=[("RandomForest",RandomForestClassifier(n_estimators=400,max_depth=16,min_samples_leaf=3,class_weight="balanced",n_jobs=-1,random_state=42))]
        if _ML_AVAILABLE.get("xgboost"):
            models.append(("XGBoost",xgb.XGBClassifier(n_estimators=350,max_depth=7,learning_rate=.04,subsample=.9,colsample_bytree=.9,eval_metric="logloss",random_state=42,n_jobs=4)))
        if _ML_AVAILABLE.get("lightgbm"):
            models.append(("LightGBM",lgb.LGBMClassifier(n_estimators=350,num_leaves=31,max_depth=9,learning_rate=.04,subsample=.9,colsample_bytree=.9,verbosity=-1,random_state=42)))
        return models

    # Expanding-window folds: initial 55%, then 15% test windows (up to 3 folds).
    n=len(X); initial=max(250,int(n*0.55)); remaining=n-initial
    test_size=max(50,int(n*0.15))
    fold_ranges=[]; train_end=initial
    while train_end<n and len(fold_ranges)<3:
        test_end=min(n,train_end+test_size)
        if test_end-train_end<30: break
        fold_ranges.append((0,train_end,train_end,test_end))
        train_end=test_end
    if not fold_ranges:
        return None

    wf={name:{"probs":[],"y":[],"metrics":[]} for name,_ in make_models()}
    for name, template in make_models():
        for tr0,tr1,te0,te1 in fold_ranges:
            try:
                scaler=RobustScaler()
                Xtr_s=scaler.fit_transform(X.iloc[tr0:tr1]); Xte_s=scaler.transform(X.iloc[te0:te1])
                m=clone(template); m.fit(Xtr_s,y.iloc[tr0:tr1])
                prob=m.predict_proba(Xte_s)[:,1]
                yt=y.iloc[te0:te1].values
                pred=(prob>=.5).astype(int)
                met={"accuracy":accuracy_score(yt,pred)*100,
                     "precision":precision_score(yt,pred,zero_division=0)*100,
                     "recall":recall_score(yt,pred,zero_division=0)*100,
                     "f1":f1_score(yt,pred,zero_division=0)*100,
                     "auc":roc_auc_score(yt,prob) if len(np.unique(yt))>1 else .5,
                     "brier":brier_score_loss(yt,prob)}
                wf[name]["probs"].extend(prob.tolist()); wf[name]["y"].extend(yt.tolist()); wf[name]["metrics"].append(met)
            except Exception as exc:
                logging.debug("walk-forward %s fold failed: %s",name,exc)

    candidates=[]
    for name,data in wf.items():
        if not data["metrics"]: continue
        mets=data["metrics"]
        avg={k:float(np.mean([m[k] for m in mets])) for k in mets[0]}
        # Ranking prioritizes OOS discrimination and penalizes poor calibration.
        rank=avg["auc"]*60 + avg["f1"]*.25 + avg["accuracy"]*.15 - avg["brier"]*20
        candidates.append((rank,name,avg,data))
    if not candidates: return None
    candidates.sort(key=lambda z:z[0],reverse=True)
    _,champion_name,champ_metrics,champ_data=candidates[0]

    # Calibrate only from genuine OOS predictions, never from in-sample data.
    calibrator=None
    try:
        probs=np.asarray(champ_data["probs"],dtype=float); yy=np.asarray(champ_data["y"],dtype=int)
        if len(probs)>=50 and len(np.unique(yy))==2 and len(np.unique(probs))>=10:
            calibrator=IsotonicRegression(y_min=0.0,y_max=1.0,out_of_bounds="clip")
            calibrator.fit(probs,yy)
    except Exception as exc:
        logging.debug("probability calibration failed: %s",exc); calibrator=None

    # Final production fit uses all historical observations only after model
    # selection/calibration has been completed from prior OOS folds.
    scaler=RobustScaler(); X_all=scaler.fit_transform(X)
    final_model=next(m for n0,m in make_models() if n0==champion_name)
    final_model.fit(X_all,y)
    artifact_dir=Path(ML_DATA_DIR)/"champions"; artifact_dir.mkdir(parents=True,exist_ok=True)
    artifact=str(artifact_dir/f"{key}_{tf}_{champion_name}.joblib")
    joblib.dump({"model":final_model,"scaler":scaler,"features":list(X.columns),
                 "calibrator":calibrator,"walk_forward":champ_metrics,
                 "folds":len(champ_data["metrics"]),"trained_samples":len(X)},artifact)

    now=int(time.time())
    with adaptive_db() as db:
        db.execute("UPDATE model_results SET is_champion=0 WHERE symbol=? AND timeframe=?",(key,tf))
        for rank,name,met,data in candidates:
            db.execute("""INSERT OR REPLACE INTO model_results
                (symbol,timeframe,model,trained_at,samples,accuracy,auc,precision_val,recall_val,f1,brier,artifact,is_champion)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",(key,tf,name,now,len(data["y"]),met["accuracy"],met["auc"],met["precision"],met["recall"],met["f1"],met["brier"],artifact if name==champion_name else "",1 if name==champion_name else 0))
    logging.info("Walk-forward ML champion %s %s/%sm: AUC=%.3f Brier=%.4f folds=%d",champion_name,key,tf,champ_metrics["auc"],champ_metrics["brier"],len(champ_data["metrics"]))
    return {"model":champion_name,**champ_metrics,"artifact":artifact,"folds":len(champ_data["metrics"]),"calibrated":calibrator is not None}

def adaptive_ml_prediction(symbol,timeframe,df):
    try:
        adaptive_init_db(); key=canonical(symbol); tf=int(timeframe)
        with adaptive_db() as db:
            row=db.execute("SELECT model,artifact,accuracy,auc,brier FROM model_results WHERE symbol=? AND timeframe=? AND is_champion=1 LIMIT 1",(key,tf)).fetchone()
        if not row: return {"prob_up":.5,"confidence":0,"model":"—"}
        pack=joblib.load(row[1]); f=adaptive_ml_features(df)
        if f is None: return {"prob_up":.5,"confidence":0,"model":row[0]}
        X=f[pack["features"]].tail(1); raw_prob=float(pack["model"].predict_proba(pack["scaler"].transform(X))[:,1][0])
        cal=pack.get("calibrator")
        prob=float(cal.predict([raw_prob])[0]) if cal is not None else raw_prob
        # Confidence is descriptive: discrimination quality × distance from 50%,
        # never a claim that the next trade has that probability of winning.
        auc=float(row[3] or .5); brier=float(row[4] or 0.25)
        quality=float(np.clip((auc-.5)*2.0,0,1))
        confidence=float(np.clip(50 + abs(prob-.5)*100*quality,0,99))
        return {"prob_up":prob,"raw_prob":raw_prob,"confidence":confidence,"auc":auc,"brier":brier,"model":row[0],"calibrated":cal is not None}
    except Exception:
        return {"prob_up":.5,"confidence":0,"model":"—"}


def adaptive_enrich_signal(base_result,symbol,timeframe=15):
    try:
        key=canonical(symbol); tf=int(timeframe); df,_=smart_ai_history(key,tf,bars=240)
        if df is None or df.empty: return base_result
        sr=adaptive_support_resistance(df); vf=adaptive_volume_features(df)
        fng=adaptive_fear_greed()
        opt=adaptive_apply_champion(df,key,tf); ml=adaptive_ml_prediction(key,tf,df)
        # Empirical components are bounded and deliberately cannot override the
        # base engine by themselves.
        bias=0.0
        if opt["signal"]>0: bias += 5
        elif opt["signal"]<0: bias -= 5
        if ml["prob_up"]>=.60: bias += 5
        elif ml["prob_up"]<=.40: bias -= 5
        if sr.get("support") and abs(float(df.close.iloc[-1])-sr["support"])/float(df.close.iloc[-1])*100<.5: bias += 3
        if sr.get("resistance") and abs(float(sr["resistance"])-float(df.close.iloc[-1]))/float(df.close.iloc[-1])*100<.5: bias -= 3
        if vf["ratio"]>=1.5: bias += 2 if vf["pressure"]>0 else -2 if vf["pressure"]<0 else 0
        score=float(np.clip(float(base_result.get("score",50))+bias,0,100))
        base_result.update({"adaptive_score":score,"optimized":opt["optimized"],"optimized_confidence":opt.get("confidence",0),
                            "optimizer_champion":opt.get("champion"),"ml_champion":ml,"fear_greed":fng,
                            "support_resistance":sr,"volume_intelligence":vf,
                            "research_status":"تاریخچه/بهینه‌سازی ذخیره‌شده" if opt["optimized"] else "نیازمند تحقیق"})
        # Do not convert a WAIT to a trade solely because of the new layer.
        if base_result.get("decision")=="WAIT": return base_result
        if score>=78: base_result["decision"]="BUY++"
        elif score>=62: base_result["decision"]="BUY"
        elif score<=22: base_result["decision"]="SELL++"
        elif score<=38: base_result["decision"]="SELL"
        else: base_result["decision"]="WAIT"
        return base_result
    except Exception as exc:
        logging.debug("adaptive_enrich_signal: %s",exc)
        return base_result


def adaptive_research_all(symbols=None,timeframes=None,years=2,train_ml=True):
    symbols=list(symbols or SYMBOLS); timeframes=list(timeframes or ADAPTIVE_RESEARCH_TIMEFRAMES)
    def runner():
        adaptive_init_db(); total=max(1,len(symbols)*len(timeframes)); done=0
        with adaptive_db() as db:
            cur=db.execute("INSERT INTO research_runs(started_at,status,message) VALUES(?,?,?)",(int(time.time()),"running","شروع تحقیق")); run_id=cur.lastrowid
        try:
            # Timeframe-major phased queue: all symbols of 1m first, then all
            # symbols of 5m, 15m, 30m, 1h, 4h and 1d.
            ordered_timeframes=[int(t) for t in timeframes]
            for phase_no,tf in enumerate(ordered_timeframes,1):
                with _adaptive_lock:
                    _adaptive_research_status.update(running=True,progress=int(done/max(1,total)*100),message=f"فاز {phase_no}/{len(ordered_timeframes)}: همه نمادها در {tf}m")
                for sym in symbols:
                    try:
                        _adaptive_set_unit_status(sym,tf,years,'updating',f'فاز {phase_no}/{len(ordered_timeframes)} — دریافت {tf}m')
                        # Bootstrap only: get enough recent candles to train quickly.
                        df=adaptive_fetch_and_store_history(sym,tf,years=years,force=False,max_requests=8)
                        audit=adaptive_history_coverage(sym,tf,years)
                        if not audit.get('trainable'):
                            _adaptive_set_unit_status(sym,tf,years,'needs_data','هنوز داده کافی برای آموزش جمع نشده',audit)
                        else:
                            adaptive_optimize_symbol_tf(sym,tf,years=years)
                            if train_ml:
                                df=adaptive_load_candles(sym,tf,audit['start'],audit['end'])
                                adaptive_train_ml_champions(sym,tf,df)
                            _adaptive_set_unit_status(sym,tf,years,'ready' if audit.get('complete') else 'ready_partial',
                                                       'آموزش انجام شد؛ تکمیل تاریخچه در پس‌زمینه',audit)
                    except Exception as exc:
                        _adaptive_set_unit_status(sym,tf,years,'error',str(exc))
                        logging.exception("adaptive research failed %s %s",sym,tf)
                    done+=1
                    with _adaptive_lock:
                        _adaptive_research_status.update(running=True,progress=int(done/total*100),message=f"{done}/{total}: {sym} در فاز {tf}m")
                # Let the GUI/network settle before starting the next timeframe.
                time.sleep(float(ADAPTIVE_TIMEFRAME_PHASE_PAUSE_SEC))
            msg="تحقیق اولیه فازبندی‌شده کامل شد؛ تکمیل تاریخچه در پس‌زمینه ادامه دارد"
            with adaptive_db() as db: db.execute("UPDATE research_runs SET finished_at=?,status=?,message=? WHERE id=?",(int(time.time()),"done",msg,run_id))
            try:
                adaptive_start_background_backfill(symbols,timeframes,years=years)
            except Exception as exc:
                logging.debug("background backfill start failed: %s",exc)
        except Exception as exc:
            with adaptive_db() as db: db.execute("UPDATE research_runs SET finished_at=?,status=?,message=? WHERE id=?",(int(time.time()),"error",str(exc),run_id))
        finally:
            with _adaptive_lock: _adaptive_research_status.update(running=False,message="پایان تحقیق",progress=100)
    threading.Thread(target=runner,daemon=True,name="AdaptiveResearch").start()


def adaptive_status_text():
    with _adaptive_lock:
        s=dict(_adaptive_research_status)
    f=adaptive_fear_greed()
    return (f"وضعیت: {s['message']} | پیشرفت: {s['progress']}% | اجرا: {'بله' if s['running'] else 'خیر'}\n"
            f"Fear & Greed: {f.get('value','—')} ({f.get('classification','—')})\n"
            f"پایگاه داده: {ADAPTIVE_DB_FILE}\n"
            f"منطق: تاریخچه کامل ذخیره‌شده + بهینه‌سازی OOS + انتخاب مدل Champion")


# Rename the original function once, then expose the enriched public entry point.
try:
    _unified_ai_decision_original = unified_ai_decision
    def unified_ai_decision(sym, timeframe=15):
        base=_unified_ai_decision_original(sym,timeframe)
        try:
            return adaptive_enrich_signal(base,sym,timeframe)
        except Exception:
            return base
except Exception:
    pass

# Persistent cache is initialized at import time; it never forces a full
# historical download. Research downloads only missing chunks when requested.
try:
    adaptive_init_db()
except Exception:
    pass





# ============================================================
# SIGNAL DISCOVERY / FUTURE TREND ENGINE v4
# ============================================================
# This layer is intentionally separate from the legacy score engine.
# It learns directional outcomes from chronological historical data,
# predicts future returns for 1h/4h/12h, and only promotes a signal
# when direction + expected move + model quality agree.
DISCOVERY_DIR = Path(APP_DATA_DIR) / "signal_discovery"
DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
DISCOVERY_MIN_ROWS = 700
DISCOVERY_MIN_CLASS_ROWS = 40
DISCOVERY_VERSION = "SD4"
DISCOVERY_HORIZONS_MIN = (60, 240, 720)
_discovery_cache = {}
_discovery_lock = threading.RLock()

def _discovery_features(df):
    try:
        if df is None or len(df) < 120:
            return None
        d = df.copy()
        c = d["close"].astype(float)
        h = d["high"].astype(float)
        l = d["low"].astype(float)
        o = d["open"].astype(float)
        v = d["volume"].astype(float)
        f = pd.DataFrame(index=d.index)

        for n in (1, 2, 3, 6, 12, 24):
            f[f"ret{n}"] = c.pct_change(n) * 100.0

        for n in (7, 14, 21, 50):
            f[f"rsi{n}"] = compute_rsi_series(d, n)

        e9 = c.ewm(span=9, adjust=False).mean()
        e21 = c.ewm(span=21, adjust=False).mean()
        e50 = c.ewm(span=50, adjust=False).mean()
        e100 = c.ewm(span=100, adjust=False).mean()
        f["ema9_dist"] = (c-e9)/c*100
        f["ema21_dist"] = (c-e21)/c*100
        f["ema50_dist"] = (c-e50)/c*100
        f["ema100_dist"] = (c-e100)/c*100
        f["ema9_21"] = (e9-e21)/c*100
        f["ema21_50"] = (e21-e50)/c*100
        f["ema50_100"] = (e50-e100)/c*100

        e12 = c.ewm(span=12, adjust=False).mean()
        e26 = c.ewm(span=26, adjust=False).mean()
        macd = e12-e26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        f["macd_pct"] = macd/c*100
        f["macd_hist_pct"] = (macd-macd_sig)/c*100

        atr = calculate_atr_series(d,14) if "calculate_atr_series" in globals() else (h-l).rolling(14).mean()
        f["atr_pct"] = atr/c*100
        f["adx"] = pd.Series(
            [calculate_adx(d.iloc[:i+1],14) if i >= 35 else np.nan for i in range(len(d))],
            index=d.index
        )

        bbmid = c.rolling(20).mean()
        bbsd = c.rolling(20).std()
        f["bb_pos"] = (c-(bbmid-2*bbsd))/(4*bbsd+1e-12)
        f["bb_width"] = (4*bbsd)/(bbmid.abs()+1e-12)*100

        hi20 = h.rolling(20).max()
        lo20 = l.rolling(20).min()
        hi50 = h.rolling(50).max()
        lo50 = l.rolling(50).min()
        f["breakout20"] = (c-hi20.shift(1))/(c.abs()+1e-12)*100
        f["breakdown20"] = (c-lo20.shift(1))/(c.abs()+1e-12)*100
        f["breakout50"] = (c-hi50.shift(1))/(c.abs()+1e-12)*100
        f["breakdown50"] = (c-lo50.shift(1))/(c.abs()+1e-12)*100

        f["range_pct"] = (h-l)/(c.abs()+1e-12)*100
        f["body_pct"] = (c-o)/(c.abs()+1e-12)*100
        f["upper_wick"] = (h-np.maximum(o,c))/(c.abs()+1e-12)*100
        f["lower_wick"] = (np.minimum(o,c)-l)/(c.abs()+1e-12)*100
        f["close_position"] = (c-l)/(h-l+1e-12)

        vm20=v.rolling(20).mean()
        vm50=v.rolling(50).mean()
        vs100=v.rolling(100).std()
        f["vol_ratio20"]=v/(vm20+1e-12)
        f["vol_ratio50"]=v/(vm50+1e-12)
        f["vol_z"]=(v-v.rolling(100).mean())/(vs100+1e-12)

        obv=(np.sign(c.diff()).fillna(0)*v).cumsum()
        f["obv_slope"] = obv.pct_change(10).replace([np.inf,-np.inf],np.nan)*100
        tp=(h+l+c)/3
        mfr=tp*v
        pos=mfr.where(tp.diff()>0,0.0).rolling(14).sum()
        neg=mfr.where(tp.diff()<0,0.0).rolling(14).sum()
        f["mfi"]=100-(100/(1+pos/(neg+1e-12)))

        vsum=v.rolling(30).sum()
        f["vwap_dist"]=(c-((tp*v).rolling(30).sum()/(vsum+1e-12)))/(c.abs()+1e-12)*100
        f["volatility20"]=c.pct_change().rolling(20).std()*100
        f["slope20"]=c.rolling(20).mean().pct_change(8)*100
        f["slope50"]=c.rolling(50).mean().pct_change(12)*100

        # Market-structure flags.
        try:
            sh, sl = detect_swing_points(d)
            bos=detect_bos_choch(d,sh,sl)
            f["bos_bull"]=1.0 if bos.get("bos_bullish") else 0.0
            f["bos_bear"]=1.0 if bos.get("bos_bearish") else 0.0
            f["choch_bull"]=1.0 if bos.get("choch_bullish") else 0.0
            f["choch_bear"]=1.0 if bos.get("choch_bearish") else 0.0
        except Exception:
            f["bos_bull"]=f["bos_bear"]=f["choch_bull"]=f["choch_bear"]=0.0

        try:
            bull,bear=detect_fvg(d)
            f["fvg_bull"]=1.0 if bull else 0.0
            f["fvg_bear"]=1.0 if bear else 0.0
        except Exception:
            f["fvg_bull"]=f["fvg_bear"]=0.0

        try:
            sweep_high,sweep_low=detect_liquidity_sweeps(d,*detect_swing_points(d))
            f["sweep_high"]=1.0 if sweep_high else 0.0
            f["sweep_low"]=1.0 if sweep_low else 0.0
        except Exception:
            f["sweep_high"]=f["sweep_low"]=0.0

        return f.replace([np.inf,-np.inf],np.nan).dropna()
    except Exception:
        return None

def _discovery_get_history(symbol, tf, bars=6000):
    key=canonical(symbol); tf=int(tf)
    try:
        d=adaptive_load_candles(key,tf)
        if d is not None and len(d)>=DISCOVERY_MIN_ROWS:
            return d.tail(bars)
    except Exception:
        pass
    try:
        d=get_candles_cached(key,tf,n=min(bars,5000),valid_symbols_map=valid_symbols_map)
        return d.copy() if d is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _discovery_horizon_bars(tf, minutes):
    return max(1,int(round(float(minutes)/max(1,int(tf)))))

def _discovery_train(symbol, tf, force=False):
    if not _ML_AVAILABLE.get("sklearn"):
        return None
    key=canonical(symbol); tf=int(tf); ck=f"{key}|{tf}"
    with _discovery_lock:
        if not force and ck in _discovery_cache:
            return _discovery_cache[ck]

    d=_discovery_get_history(key,tf)
    if d is None or len(d)<DISCOVERY_MIN_ROWS:
        return None
    f=_discovery_features(d)
    if f is None or len(f)<DISCOVERY_MIN_ROWS//2:
        return None

    # Align labels with the exact feature timestamps.  Labels are 3-class:
    # LONG, NEUTRAL, SHORT. Threshold is adaptive to current volatility.
    atrpct=(calculate_atr_series(d,14)/d["close"]*100 if "calculate_atr_series" in globals()
            else (d["high"]-d["low"]).rolling(14).mean()/d["close"]*100)
    result={"symbol":key,"timeframe":tf,"models":{},"trained_at":int(time.time()),
            "samples":len(f),"version":DISCOVERY_VERSION}
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

    for mins in DISCOVERY_HORIZONS_MIN:
        hb=_discovery_horizon_bars(tf,mins)
        if hb < 1:
            continue
        future=d["close"].shift(-hb)
        ret=(future/d["close"]-1)*100
        threshold=np.maximum(0.30, atrpct.fillna(0)*0.55)
        lab=pd.Series(0,index=d.index,dtype=int)
        lab[ret>=threshold]=1
        lab[ret<=-threshold]=-1
        common=f.index.intersection(lab.dropna().index)
        X=f.loc[common]
        y=lab.loc[common]
        valid=y.notna()
        X=X.loc[valid]; y=y.loc[valid]
        if len(X)<DISCOVERY_MIN_ROWS//2 or y.nunique()<2:
            continue
        cut=int(len(X)*0.72)
        Xtr,Xte=X.iloc[:cut],X.iloc[cut:]
        ytr,yte=y.iloc[:cut],y.iloc[cut:]
        if len(Xte)<80 or ytr.nunique()<2:
            continue
        scaler=RobustScaler()
        Xtr_s=scaler.fit_transform(Xtr); Xte_s=scaler.transform(Xte)
        model=RandomForestClassifier(
            n_estimators=450,max_depth=14,min_samples_leaf=4,
            class_weight="balanced_subsample",n_jobs=-1,random_state=1000+hb
        )
        try:
            model.fit(Xtr_s,ytr)
            prob=model.predict_proba(Xte_s)
            pred=model.predict(Xte_s)
            classes=list(model.classes_)
            cls_to_col={int(c):i for i,c in enumerate(classes)}
            p_up=prob[:,cls_to_col[1]] if 1 in cls_to_col else np.zeros(len(prob))
            p_dn=prob[:,cls_to_col[-1]] if -1 in cls_to_col else np.zeros(len(prob))
            # OOS metrics use one-vs-rest, avoiding a fake binary "up only" label.
            y_up=(yte.values==1).astype(int)
            auc=float(roc_auc_score(y_up,p_up)) if len(np.unique(y_up))==2 else .5
            acc=float(accuracy_score(yte,pred)*100)
            f1=float(f1_score(yte,pred,average="macro",zero_division=0)*100)
            artifact=DISCOVERY_DIR/f"{key}_{tf}_{mins}m.joblib"
            joblib.dump({"model":model,"scaler":scaler,"features":list(X.columns),
                         "classes":classes,"horizon_minutes":mins,
                         "threshold_mode":"0.55ATR_or_0.30pct",
                         "oos_accuracy":acc,"oos_auc":auc,"oos_f1":f1,
                         "samples":len(X),"trained_at":int(time.time()),
                         "version":DISCOVERY_VERSION},artifact)
            result["models"][str(mins)]={
                "artifact":str(artifact),"accuracy":acc,"auc":auc,"f1":f1,
                "samples":len(X),"classes":classes
            }
        except Exception as exc:
            logging.debug("signal discovery train %s %sm: %s",key,mins,exc)

    # Separate multi-output regression: signed future return, not absolute move.
    regs={}
    for mins in DISCOVERY_HORIZONS_MIN:
        hb=_discovery_horizon_bars(tf,mins)
        future=d["close"].shift(-hb)
        yret=(future/d["close"]-1)*100
        common=f.index.intersection(yret.dropna().index)
        X=f.loc[common]; y=yret.loc[common]
        if len(X)<DISCOVERY_MIN_ROWS//2:
            continue
        cut=int(len(X)*0.72)
        Xtr,Xte=X.iloc[:cut],X.iloc[cut:]; ytr,yte=y.iloc[:cut],y.iloc[cut:]
        try:
            scaler=RobustScaler()
            Xtr_s=scaler.fit_transform(Xtr); Xte_s=scaler.transform(Xte)
            reg=RandomForestRegressor(n_estimators=350,max_depth=14,min_samples_leaf=4,n_jobs=-1,random_state=2000+hb)
            reg.fit(Xtr_s,ytr)
            pred=reg.predict(Xte_s)
            mae=float(np.mean(np.abs(pred-yte.values)))
            artifact=DISCOVERY_DIR/f"{key}_{tf}_{mins}m_reg.joblib"
            joblib.dump({"model":reg,"scaler":scaler,"features":list(X.columns),
                         "horizon_minutes":mins,"oos_mae":mae,
                         "samples":len(X),"trained_at":int(time.time()),
                         "version":DISCOVERY_VERSION},artifact)
            regs[str(mins)]={"artifact":str(artifact),"mae":mae,"samples":len(X)}
        except Exception as exc:
            logging.debug("signal discovery regression %s %sm: %s",key,mins,exc)
    result["regressors"]=regs

    if result["models"] or result["regressors"]:
        with _discovery_lock:
            _discovery_cache[ck]=result
        # Also persist a compact index.
        try:
            (DISCOVERY_DIR/f"{key}_{tf}_summary.json").write_text(
                json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        except Exception:
            pass
        return result
    return None

def _discovery_load(symbol,tf):
    key=canonical(symbol); tf=int(tf); ck=f"{key}|{tf}"
    with _discovery_lock:
        if ck in _discovery_cache:
            return _discovery_cache[ck]
    summary=DISCOVERY_DIR/f"{key}_{tf}_summary.json"
    if not summary.exists():
        return None
    try:
        r=json.loads(summary.read_text(encoding="utf-8"))
        with _discovery_lock:
            _discovery_cache[ck]=r
        return r
    except Exception:
        return None

def signal_discovery_predict(symbol, tf=15, train_if_missing=False):
    """Return a real directional forecast. No absolute-positive fake move."""
    key=canonical(symbol); tf=int(tf)
    d=_discovery_get_history(key,tf,bars=1000)
    if d is None or d.empty:
        return {"ready":False,"direction":"WAIT","prob_long":0.0,"prob_short":0.0,
                "prob_neutral":1.0,"quality":0.0,"reason":"History ناکافی"}
    f=_discovery_features(d)
    if f is None or f.empty:
        return {"ready":False,"direction":"WAIT","prob_long":0.0,"prob_short":0.0,
                "prob_neutral":1.0,"quality":0.0,"reason":"Feature ناکافی"}
    r=_discovery_load(key,tf)
    if r is None and train_if_missing:
        r=_discovery_train(key,tf)
    if not r:
        return {"ready":False,"direction":"WAIT","prob_long":0.0,"prob_short":0.0,
                "prob_neutral":1.0,"quality":0.0,"reason":"مدل آموزش‌دیده برای این نماد/تایم‌فریم وجود ندارد"}

    x=f.tail(1)
    horizons={}
    for mins,info in r.get("models",{}).items():
        try:
            pack=joblib.load(info["artifact"])
            xs=pack["scaler"].transform(x[pack["features"]])
            pr=pack["model"].predict_proba(xs)[0]
            classes=[int(z) for z in pack["classes"]]
            probs={c:float(pr[i]) for i,c in enumerate(classes)}
            pu=probs.get(1,0.0); pdn=probs.get(-1,0.0); pn=probs.get(0,0.0)
            reg=None
            ri=r.get("regressors",{}).get(str(mins))
            if ri:
                rp=joblib.load(ri["artifact"])
                reg=float(rp["model"].predict(rp["scaler"].transform(x[rp["features"]]))[0])
            horizons[str(mins)]={"long":pu,"short":pdn,"neutral":pn,"expected_return":reg,
                                 "oos_auc":float(info.get("auc",.5)),
                                 "oos_accuracy":float(info.get("accuracy",0)),
                                 "oos_f1":float(info.get("f1",0))}
        except Exception as exc:
            logging.debug("discovery prediction %s %s: %s",key,mins,exc)

    if not horizons:
        return {"ready":False,"direction":"WAIT","prob_long":0.0,"prob_short":0.0,
                "prob_neutral":1.0,"quality":0.0,"reason":"مدل قابل بارگذاری نیست"}

    # Primary decision is the shortest horizon that exists; longer horizons
    # are used as trend confirmation, not mixed into one arbitrary score.
    ordered=sorted(horizons.items(),key=lambda z:int(z[0]))
    primary=ordered[0][1]
    pl=primary["long"]; ps=primary["short"]; pn=primary["neutral"]
    direction="LONG" if pl>=ps and pl>=pn else "SHORT" if ps>pl and ps>=pn else "WAIT"

    # Trend agreement across available horizons.
    signs=[]
    for _,h in ordered:
        er=h.get("expected_return")
        if er is not None:
            signs.append(1 if er>0 else -1 if er<0 else 0)
    if signs:
        if direction=="LONG" and sum(s>0 for s in signs) < max(1,math.ceil(len(signs)*.67)):
            direction="WAIT"
        elif direction=="SHORT" and sum(s<0 for s in signs) < max(1,math.ceil(len(signs)*.67)):
            direction="WAIT"

    quality=float(np.clip(
        max(pl,ps)*55 +
        (1-abs(pl-ps))*0 +
        np.mean([h["oos_auc"] for h in horizons.values()])*25 +
        np.mean([h["oos_f1"] for h in horizons.values()])*.20,0,100))
    # Strongly penalize ambiguous probability.
    if max(pl,ps)<0.60 or abs(pl-ps)<0.12:
        quality*=0.65

    return {
        "ready":True,"direction":direction,
        "prob_long":pl,"prob_short":ps,"prob_neutral":pn,
        "quality":float(np.clip(quality,0,100)),
        "horizons":horizons,
        "primary_expected_return":primary.get("expected_return"),
        "reason":"هم‌جهتی مدل + روند چندافق زمانی" if direction!="WAIT" else "تعارض/ابهام مدل‌ها"
    }

def signal_discovery_decision(sym, timeframe=15):
    """Merge legacy analysis with the new evidence-based discovery layer."""
    tf=int(timeframe); key=canonical(sym)
    try:
        base=_unified_ai_decision_adaptive_legacy(key,tf)
    except Exception:
        base={"symbol":key,"decision":"WAIT","score":50.0,"confidence":0.0,
              "price":safe_float(valid_symbols_map.get(key))}
    pred=signal_discovery_predict(key,tf,train_if_missing=False)

    if not pred.get("ready"):
        base["discovery"]=pred
        base["discovery_ready"]=False
        return base

    pl=float(pred.get("prob_long",0)); ps=float(pred.get("prob_short",0))
    q=float(pred.get("quality",0))
    er=pred.get("primary_expected_return")
    # Discovery is now the directional authority. Legacy engine can veto
    # only when it strongly disagrees; it cannot manufacture a direction.
    legacy_score=float(base.get("score",50))
    legacy_dir="LONG" if legacy_score>=62 else "SHORT" if legacy_score<=38 else "WAIT"
    direction=pred["direction"]
    if direction=="LONG" and pl>=.67 and q>=60:
        decision="BUY++" if pl>=.78 and q>=75 else "BUY"
    elif direction=="SHORT" and ps>=.67 and q>=60:
        decision="SELL++" if ps>=.78 and q>=75 else "SELL"
    else:
        decision="WAIT"

    # If the legacy engine is strongly opposite, reduce confidence rather
    # than silently flipping the discovery result.
    conflict=(direction=="LONG" and legacy_dir=="SHORT") or (direction=="SHORT" and legacy_dir=="LONG")
    if conflict:
        q*=0.72
        if q<60:
            decision="WAIT"

    base.update({
        "decision":decision,
        "signal_state":("🟢 BUY++" if decision=="BUY++" else "🟩 BUY" if decision=="BUY"
                        else "🔴 SELL++" if decision=="SELL++" else "🟥 SELL" if decision=="SELL" else "⚪ WAIT"),
        "discovery":pred,
        "discovery_ready":True,
        "discovery_direction":direction,
        "prob_long":pl*100,
        "prob_short":ps*100,
        "prob_neutral":float(pred.get("prob_neutral",0))*100,
        "signal_quality":q,
        "trend_1h":pred.get("horizons",{}).get("60",{}).get("expected_return"),
        "trend_4h":pred.get("horizons",{}).get("240",{}).get("expected_return"),
        "trend_12h":pred.get("horizons",{}).get("720",{}).get("expected_return"),
        "discovery_reason":pred.get("reason",""),
    })
    if er is not None:
        base["predicted_move_pct"]=abs(float(er))
        base["predicted_signed_move_pct"]=float(er)
        p=float(base.get("price") or 0)
        if p>0:
            base["discovery_target_price"]=p*(1+float(er)/100)
    base["confidence"]=float(np.clip(max(float(base.get("confidence",0)),q),0,99))
    return base

# Preserve the previous adaptive entry point for compatibility.
try:
    _unified_ai_decision_adaptive_legacy = unified_ai_decision
except Exception:
    _unified_ai_decision_adaptive_legacy = None

# Final public decision function used by AI/smart scanners.
def unified_ai_decision(sym, timeframe=15):
    try:
        return signal_discovery_decision(sym,timeframe)
    except Exception as exc:
        logging.debug("signal_discovery_decision failed %s: %s",sym,exc)
        return _unified_ai_decision_adaptive_legacy(sym,timeframe) if _unified_ai_decision_adaptive_legacy else {"symbol":canonical(sym),"decision":"WAIT","score":50.0}

# Replace the old research launcher with a timeframe-phased research launcher:
# 1m for every symbol -> 5m for every symbol -> 15m -> 30m -> 1h -> 4h -> 1d.
# This ordering is intentional: it gives a complete market-wide pass per TF,
# reduces context switching, and makes the GUI progress meaningful.
def signal_discovery_research(symbols=None,timeframes=None,force=False):
    symbols=list(symbols or SYMBOLS)
    timeframes=list(timeframes or ADAPTIVE_RESEARCH_TIMEFRAMES)
    def worker():
        total=max(1,len(symbols)*len(timeframes)); done=0
        for tf in timeframes:
            for sym in symbols:
                try:
                    adaptive_fetch_and_store_history(sym,tf,years=2,force=False)
                    _discovery_train(sym,tf,force=force)
                except Exception as exc:
                    logging.debug("discovery research failed %s/%s: %s",sym,tf,exc)
                done+=1
                try:
                    with _adaptive_lock:
                        _adaptive_research_status.update(
                            running=True,progress=int(done/total*100),
                            message=f"Discovery {done}/{total}: {sym} {tf}m")
                except Exception:
                    pass
        try:
            with _adaptive_lock:
                _adaptive_research_status.update(running=False,progress=100,message="تحقیق کشف سیگنال پایان یافت")
        except Exception:
            pass
    threading.Thread(target=worker,daemon=True,name="SignalDiscoveryResearch").start()

# Existing research button entry point: discovery research is the canonical path.
def adaptive_research_all_legacy_complete(symbols=None,timeframes=None,years=2,train_ml=True):
    """Complete 2-year research: each method independently per TF, then best-method combination."""
    symbols=list(symbols or SYMBOLS)
    timeframes=list(timeframes or (5,15,60,240,1440))
    def runner():
        # The first long research run explicitly asks where its persistent files
        # should live. Subsequent runs reuse that location until the user changes it.
        if not _deep_ensure_storage_location():
            logging.info("Research storage location not selected; using existing/default location: %s", DEEP_RESEARCH_DIR)
        adaptive_init_db()
        _deep_write_master_index({"run_started_at":time.time(),"years":years,"symbols":len(symbols),"timeframes":list(timeframes)})
        total=max(1,len(symbols)*len(timeframes)); done=0
        reports_dir=Path(APP_DATA_DIR)/"deep_method_research"; reports_dir.mkdir(parents=True,exist_ok=True)
        try:
            for sym in symbols:
                for tf in timeframes:
                    try:
                        # Download + exact audit FIRST. Never train/optimize silently on a partial window.
                        df=adaptive_fetch_and_store_history(sym,tf,years=years,force=False)
                        audit=adaptive_history_coverage(sym,tf,years)
                        adaptive_save_history_audit(audit)
                        if audit.get("complete"):
                            df=adaptive_load_candles(sym,tf,audit["start"],audit["end"])
                        self_audit = (f"{canonical(sym)} {tf}m | {audit['count']}/{audit['expected']} candles "
                                      f"| coverage={audit['coverage_pct']:.2f}% | gaps={audit['gap_count']} "
                                      f"| {'COMPLETE' if audit['complete'] else 'INCOMPLETE'}")
                        logging.info(self_audit)
                        if not audit["complete"]:
                            with _adaptive_lock:
                                _adaptive_research_status.update(message=f"تاریخچه ناقص: {self_audit}")
                            done += 1
                            continue
                        # Only after exact history verification may optimization/ML start.
                        adaptive_optimize_symbol_tf(sym,tf,years=years)
                        try:
                            cp0=_deep_load_checkpoint(sym,tf,years=years)
                            if cp0.get("methods"):
                                logging.info("Research resume %s/%sm: %d checkpointed units", canonical(sym), tf, len(cp0.get("methods",{})))
                        except Exception:
                            pass
                        if len(df)>=250:
                            cp = _deep_load_checkpoint(sym, tf, years=years)
                            def _save_method(_method, _payload, _all):
                                cp["methods"] = dict(_all)
                                cp["last_completed"] = _method
                                cp["status"] = "running"
                                _deep_save_checkpoint(sym, tf, cp, years=years)
                            research=deep_research_dataframe(df, checkpoint=cp, checkpoint_save=_save_method)
                            payload={"symbol":canonical(sym),"timeframe":tf,"years":years,"research_schema":DEEP_RESEARCH_SCHEMA_VERSION,"fingerprint":_deep_research_fingerprint(years),"research":research,"created_at":time.time()}
                            report_path=reports_dir/f"{canonical(sym)}_{tf}m.json"
                            report_tmp=report_path.with_suffix(report_path.suffix+".tmp")
                            # Keep the previous completed report before replacing it.
                            _deep_backup_file(report_path)
                            report_tmp.write_text(json.dumps(payload,ensure_ascii=False,default=str,indent=2),encoding="utf-8")
                            os.replace(str(report_tmp),str(report_path))
                            _deep_write_master_index({"last_completed_symbol":canonical(sym),"last_completed_timeframe":tf,"last_report":str(report_path)})
                            cp["methods"] = dict(research)
                            cp["status"] = "done"
                            cp["completed_at"] = time.time()
                            _deep_save_checkpoint(sym, tf, cp, years=years)
                            if train_ml:
                                try: adaptive_train_ml_champions(sym,tf,df)
                                except Exception: pass
                    except Exception as exc:
                        logging.exception("complete research failed %s %s: %s",sym,tf,exc)
                    done+=1
                    with _adaptive_lock:
                        _adaptive_research_status.update(running=True,progress=int(done/total*100),message=f"روش‌ها مستقل + ترکیب {done}/{total}: {sym} {tf}m")
            with _adaptive_lock: _adaptive_research_status.update(running=False,progress=100,message="تحقیق کامل روش‌ها + ترکیب بهترین‌ها پایان یافت")
        except Exception as exc:
            with _adaptive_lock: _adaptive_research_status.update(running=False,message=f"خطای تحقیق: {exc}")
    threading.Thread(target=runner,daemon=True,name="CompleteMethodResearch").start()


# ============================================================
# EVIDENCE SIGNAL LEDGER V5
# ============================================================
# هدف:
# 1) ثبت snapshot کامل هر Signal Discovery واقعی
# 2) اعتبارسنجی خودکار در افق‌های 1h/4h/12h
# 3) MFE / MAE و WIN / LOSS / FAKE / NEUTRAL
# 4) استخراج تاریخی سیگنال‌ها و سنجش Rule / Combination
# 5) محاسبه Precision / Win Rate / PF / Expectancy و OOS diagnostics
# 6) نگهداری نسخه مدل/قوانین برای اینکه تغییرات قابل اندازه‌گیری باشند
#
# این لایه هیچ معامله‌ای را خودکار فعال نمی‌کند و به‌تنهایی
# ادعای سود یا پیش‌بینی قطعی بازار ندارد.
from pathlib import Path as _EvidencePath
import itertools as _evidence_itertools

EVIDENCE_DIR = _EvidencePath(APP_DATA_DIR) / "evidence_signal_engine"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DB = EVIDENCE_DIR / "signal_ledger_v6.sqlite3"
EVIDENCE_VERSION = "ESL6.0"
EVIDENCE_HORIZONS = (60, 240, 720)
EVIDENCE_MIN_MOVE_PCT = 0.30
EVIDENCE_ATR_MULT = 0.55
EVIDENCE_MIN_RULE_SAMPLES = 20
_EVIDENCE_LOCK = threading.RLock()
_EVIDENCE_VALIDATOR_STARTED = False
_EVIDENCE_LAST_SCAN = 0.0


def evidence_db():
    db = sqlite3.connect(str(EVIDENCE_DB), timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    return db


def evidence_init_db():
    with evidence_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS signal_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_uid TEXT UNIQUE,
            created_at REAL NOT NULL,
            symbol TEXT NOT NULL,
            timeframe INTEGER NOT NULL,
            direction TEXT NOT NULL,
            decision TEXT,
            entry REAL NOT NULL,
            score REAL,
            confidence REAL,
            probability REAL,
            prob_long REAL,
            prob_short REAL,
            prob_neutral REAL,
            quality REAL,
            predicted_move_pct REAL,
            predicted_signed_move_pct REAL,
            target_price REAL,
            discovery_target_price REAL,
            trend_1h REAL,
            trend_4h REAL,
            trend_12h REAL,
            regime TEXT,
            legacy_score REAL,
            mtf_align REAL,
            rules_json TEXT,
            features_json TEXT,
            reasons_json TEXT,
            model_version TEXT,
            engine_version TEXT,
            status TEXT DEFAULT 'OPEN'
        );

        CREATE TABLE IF NOT EXISTS signal_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_uid TEXT NOT NULL,
            horizon_min INTEGER NOT NULL,
            evaluated_at REAL,
            future_price REAL,
            signed_return_pct REAL,
            mfe_pct REAL,
            mae_pct REAL,
            threshold_pct REAL,
            result TEXT,
            bars_observed INTEGER,
            FOREIGN KEY(signal_uid) REFERENCES signal_ledger(signal_uid),
            UNIQUE(signal_uid, horizon_min)
        );

        CREATE TABLE IF NOT EXISTS historical_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe INTEGER NOT NULL,
            ts REAL NOT NULL,
            direction TEXT NOT NULL,
            entry REAL NOT NULL,
            score REAL,
            rules_json TEXT,
            features_json TEXT,
            horizon_min INTEGER NOT NULL,
            future_price REAL,
            signed_return_pct REAL,
            mfe_pct REAL,
            mae_pct REAL,
            threshold_pct REAL,
            result TEXT,
            target_hit INTEGER DEFAULT 0,
            stop_hit INTEGER DEFAULT 0,
            ambiguous_path INTEGER DEFAULT 0,
            bars_to_target INTEGER,
            bars_to_stop INTEGER,
            bars_to_mfe INTEGER,
            bars_to_mae INTEGER,
            label_confidence REAL DEFAULT 1.0
        );

        CREATE TABLE IF NOT EXISTS rule_stats (
            rule_name TEXT NOT NULL,
            direction TEXT NOT NULL,
            horizon_min INTEGER NOT NULL,
            samples INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            fake INTEGER NOT NULL,
            neutral INTEGER NOT NULL,
            win_rate REAL,
            precision_val REAL,
            avg_return REAL,
            expectancy REAL,
            profit_factor REAL,
            avg_mfe REAL,
            avg_mae REAL,
            updated_at REAL,
            PRIMARY KEY(rule_name, direction, horizon_min)
        );

        CREATE TABLE IF NOT EXISTS combo_stats (
            combo_key TEXT NOT NULL,
            direction TEXT NOT NULL,
            horizon_min INTEGER NOT NULL,
            samples INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            fake INTEGER NOT NULL,
            neutral INTEGER NOT NULL,
            win_rate REAL,
            precision_val REAL,
            avg_return REAL,
            expectancy REAL,
            profit_factor REAL,
            avg_mfe REAL,
            avg_mae REAL,
            updated_at REAL,
            PRIMARY KEY(combo_key, direction, horizon_min)
        );

        CREATE TABLE IF NOT EXISTS research_runs (
            run_id TEXT PRIMARY KEY,
            started_at REAL,
            finished_at REAL,
            status TEXT,
            symbols INTEGER,
            timeframes INTEGER,
            signals INTEGER,
            message TEXT
        );

        CREATE TABLE IF NOT EXISTS engine_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            measured_at REAL,
            engine_version TEXT,
            model_version TEXT,
            dataset TEXT,
            samples INTEGER,
            signals INTEGER,
            wins INTEGER,
            losses INTEGER,
            fake INTEGER,
            neutral INTEGER,
            win_rate REAL,
            precision_val REAL,
            avg_return REAL,
            expectancy REAL,
            profit_factor REAL,
            max_drawdown REAL,
            note TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_ledger_pending
            ON signal_ledger(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_outcome_uid
            ON signal_outcomes(signal_uid);
        CREATE INDEX IF NOT EXISTS idx_hist_run
            ON historical_signals(run_id);
        CREATE INDEX IF NOT EXISTS idx_hist_rule
            ON historical_signals(symbol, timeframe, horizon_min);
        """)
        # Backward-compatible migration for databases created by ESL5.x.
        existing = {r[1] for r in db.execute("PRAGMA table_info(historical_signals)").fetchall()}
        migrations = {
            "target_hit": "ALTER TABLE historical_signals ADD COLUMN target_hit INTEGER DEFAULT 0",
            "stop_hit": "ALTER TABLE historical_signals ADD COLUMN stop_hit INTEGER DEFAULT 0",
            "ambiguous_path": "ALTER TABLE historical_signals ADD COLUMN ambiguous_path INTEGER DEFAULT 0",
            "bars_to_target": "ALTER TABLE historical_signals ADD COLUMN bars_to_target INTEGER",
            "bars_to_stop": "ALTER TABLE historical_signals ADD COLUMN bars_to_stop INTEGER",
            "bars_to_mfe": "ALTER TABLE historical_signals ADD COLUMN bars_to_mfe INTEGER",
            "bars_to_mae": "ALTER TABLE historical_signals ADD COLUMN bars_to_mae INTEGER",
            "label_confidence": "ALTER TABLE historical_signals ADD COLUMN label_confidence REAL DEFAULT 1.0",
        }
        for col, sql in migrations.items():
            if col not in existing:
                db.execute(sql)
evidence_init_db()


def _evidence_json(value):
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


def _evidence_float(value, default=0.0):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _evidence_result(direction, signed_return, threshold, target_hit=False,
                      stop_hit=False, ambiguous=False):
    """Conservative path-aware label.

    A close-only label can hide a path that first hit the stop and later recovered.
    Here, target/stop are evaluated on the path. If both are touched inside the
    same candle, the order is unknowable from OHLC alone, so the sample is marked
    NEUTRAL with reduced label confidence rather than inventing an intrabar order.
    """
    sr = _evidence_float(signed_return)
    th = max(EVIDENCE_MIN_MOVE_PCT, _evidence_float(threshold, EVIDENCE_MIN_MOVE_PCT))
    if ambiguous:
        return "NEUTRAL"
    if target_hit and not stop_hit:
        return "WIN"
    if stop_hit and not target_hit:
        return "LOSS"
    if sr >= th:
        return "WIN"
    if sr <= -th:
        return "LOSS"
    return "NEUTRAL"


def _evidence_path_metrics(df, start_i, horizon_bars, direction):
    """Evaluate the future path with conservative first-touch semantics.

    The signal candle itself is never included in the outcome path. Target and
    stop are symmetric and ATR-derived. If both are touched on one OHLC candle,
    the result is deliberately ambiguous instead of assuming an intrabar order.
    """
    end_i = min(len(df) - 1, start_i + max(1, int(horizon_bars)))
    if end_i <= start_i:
        return None
    entry = _evidence_float(df["close"].iloc[start_i])
    if entry <= 0:
        return None
    future = df.iloc[start_i + 1:end_i + 1]
    if future.empty:
        return None

    atr = 0.0
    try:
        w = df.iloc[max(0, start_i - 80):start_i + 1]
        atr_abs = abs(float(calculate_atr(w, 14)))
        atr = atr_abs / entry * 100.0
    except Exception:
        atr = 0.0
    threshold = max(EVIDENCE_MIN_MOVE_PCT, atr * EVIDENCE_ATR_MULT)

    # Target/stop are intentionally separated from the final close label.
    target_pct = threshold
    stop_pct = threshold * 0.90
    target_hit = stop_hit = False
    ambiguous = False
    bars_to_target = bars_to_stop = None
    for n, (_, row) in enumerate(future.iterrows(), start=1):
        hi = _evidence_float(row.get("high"), entry)
        lo = _evidence_float(row.get("low"), entry)
        if direction == "LONG":
            hit_t = hi >= entry * (1.0 + target_pct / 100.0)
            hit_s = lo <= entry * (1.0 - stop_pct / 100.0)
        else:
            hit_t = lo <= entry * (1.0 - target_pct / 100.0)
            hit_s = hi >= entry * (1.0 + stop_pct / 100.0)
        if hit_t and hit_s:
            ambiguous = True
            if bars_to_target is None: bars_to_target = n
            if bars_to_stop is None: bars_to_stop = n
            break
        if hit_t and not target_hit:
            target_hit = True
            bars_to_target = n
        if hit_s and not stop_hit:
            stop_hit = True
            bars_to_stop = n
        if target_hit or stop_hit:
            # First decisive touch is enough for the primary label.
            break

    close = _evidence_float(future["close"].iloc[-1])
    signed = ((close - entry) / entry * 100.0) if direction == "LONG" else ((entry - close) / entry * 100.0)
    if direction == "LONG":
        mfe_series = (future["high"].astype(float) - entry) / entry * 100.0
        mae_series = (future["low"].astype(float) - entry) / entry * 100.0
    else:
        mfe_series = (entry - future["low"].astype(float)) / entry * 100.0
        mae_series = (entry - future["high"].astype(float)) / entry * 100.0
    mfe = float(mfe_series.max())
    mae = float(mae_series.min())
    bars_to_mfe = int(mfe_series.values.argmax()) + 1
    bars_to_mae = int(mae_series.values.argmin()) + 1

    result = _evidence_result(direction, signed, threshold, target_hit, stop_hit, ambiguous)
    # A favorable excursion followed by a stop/negative close is explicitly FAKE.
    if not ambiguous and result == "LOSS" and mfe >= target_pct * 0.85:
        result = "FAKE"
    # Confidence is lower when the path was too close to both barriers or label
    # depends mainly on the final close rather than a decisive first touch.
    if ambiguous:
        label_conf = 0.35
    elif target_hit or stop_hit:
        label_conf = 1.0
    else:
        margin = abs(signed) / max(threshold, 1e-9)
        label_conf = float(np.clip(0.55 + 0.25 * min(margin, 1.0), 0.55, 0.80))

    return {
        "future_price": close,
        "signed_return_pct": signed,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "threshold_pct": threshold,
        "result": result,
        "bars_observed": len(future),
        "target_hit": int(target_hit),
        "stop_hit": int(stop_hit),
        "ambiguous_path": int(ambiguous),
        "bars_to_target": bars_to_target,
        "bars_to_stop": bars_to_stop,
        "bars_to_mfe": bars_to_mfe,
        "bars_to_mae": bars_to_mae,
        "label_confidence": label_conf,
        "evaluated_at": time.time(),
    }

def _evidence_extract_rules(result):
    rules = list(result.get("matched_rules_names") or result.get("matched_rules") or [])
    # Discovery/legacy engines may not expose every component as a named rule.
    # Build explicit, reproducible rule names from the snapshot fields.
    score = _evidence_float(result.get("score"))
    prob_l = _evidence_float(result.get("prob_long"))
    prob_s = _evidence_float(result.get("prob_short"))
    vol = _evidence_float(
        (result.get("volume_intelligence") or {}).get("ratio"),
        0.0
    )
    if prob_l >= 67:
        rules.append("Discovery-Long>=67")
    if prob_s >= 67:
        rules.append("Discovery-Short>=67")
    if score >= 62:
        rules.append("Legacy-Bull>=62")
    if score <= 38:
        rules.append("Legacy-Bear<=38")
    if vol >= 1.5:
        rules.append("VolumeExpansion>=1.5")
    if result.get("optimized"):
        rules.append("OptimizerChampion")
    if result.get("ml_champion"):
        rules.append("MLChampion")
    # Preserve order and uniqueness.
    out, seen = [], set()
    for r in rules:
        s = str(r).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:12]


def _evidence_snapshot_features(result):
    keep = (
        "score", "confidence", "probability", "prob_long", "prob_short",
        "prob_neutral", "signal_quality", "mtf_align", "regime",
        "trend_1h", "trend_4h", "trend_12h", "predicted_move_pct",
        "predicted_signed_move_pct", "adaptive_score", "indicator_confluence",
        "imbalance_score", "adx", "rsi", "macd", "volume_ratio",
        "vol_ratio", "bb_squeeze", "bos_bullish", "bos_bearish",
        "choch_bullish", "choch_bearish"
    )
    return {k: result.get(k) for k in keep if k in result}


def evidence_record_signal(result, timeframe=15):
    """
    Persist a live signal snapshot. WAIT is intentionally not recorded as a
    trade signal unless discovery produced a directional candidate.
    """
    try:
        if not isinstance(result, dict):
            return None
        decision = str(result.get("decision", "WAIT"))
        if decision not in {"BUY", "BUY++", "SELL", "SELL++"}:
            return None
        direction = "LONG" if decision.startswith("BUY") else "SHORT"
        symbol = canonical(result.get("symbol", ""))
        entry = _evidence_float(result.get("price") or result.get("entry"))
        if not symbol or entry <= 0:
            return None
        created = _evidence_float(result.get("signal_timestamp") or time.time(), time.time())
        uid = f"{symbol}-{int(created*1000)}-{direction}-{EVIDENCE_VERSION}"
        rules = _evidence_extract_rules(result)
        model_version = (
            (result.get("discovery") or {}).get("version")
            or (result.get("ml_champion") or {}).get("model")
            or "fallback"
        )
        reasons = {
            "discovery_reason": result.get("discovery_reason"),
            "rules": rules,
            "research_status": result.get("research_status"),
        }
        with evidence_db() as db:
            db.execute("""
                INSERT OR IGNORE INTO signal_ledger(
                    signal_uid,created_at,symbol,timeframe,direction,decision,entry,
                    score,confidence,probability,prob_long,prob_short,prob_neutral,
                    quality,predicted_move_pct,predicted_signed_move_pct,target_price,
                    discovery_target_price,trend_1h,trend_4h,trend_12h,regime,
                    legacy_score,mtf_align,rules_json,features_json,reasons_json,
                    model_version,engine_version,status
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                uid, created, symbol, int(timeframe), direction, decision, entry,
                _evidence_float(result.get("score"), 50),
                _evidence_float(result.get("confidence")),
                _evidence_float(result.get("probability")),
                _evidence_float(result.get("prob_long")) / 100.0,
                _evidence_float(result.get("prob_short")) / 100.0,
                _evidence_float(result.get("prob_neutral")) / 100.0,
                _evidence_float(result.get("signal_quality") or result.get("confidence")),
                _evidence_float(result.get("predicted_move_pct")),
                _evidence_float(result.get("predicted_signed_move_pct")),
                _evidence_float(result.get("target_price")),
                _evidence_float(result.get("discovery_target_price")),
                result.get("trend_1h"),
                result.get("trend_4h"),
                result.get("trend_12h"),
                result.get("regime"),
                _evidence_float(result.get("score"), 50),
                _evidence_float(result.get("mtf_align"), 0.5),
                _evidence_json(rules),
                _evidence_json(_evidence_snapshot_features(result)),
                _evidence_json(reasons),
                str(model_version),
                EVIDENCE_VERSION,
                "OPEN"
            ))
        return uid
    except Exception as exc:
        logging.debug("evidence_record_signal failed: %s", exc)
        return None


def evidence_validate_pending(max_signals=250):
    """
    Validate due live signals from fresh candles. Each signal receives
    independent 1h/4h/12h outcomes. A signal is COMPLETE only when all
    available requested horizons are evaluated.
    """
    done = 0
    try:
        now = time.time()
        with evidence_db() as db:
            rows = db.execute("""
                SELECT signal_uid,created_at,symbol,timeframe,direction,entry
                FROM signal_ledger
                WHERE status='OPEN'
                ORDER BY created_at ASC
                LIMIT ?
            """, (int(max_signals),)).fetchall()

        for uid, created, symbol, tf, direction, entry in rows:
            try:
                d = _discovery_get_history(symbol, int(tf), bars=3000)
                if d is None or d.empty:
                    continue
                for horizon in EVIDENCE_HORIZONS:
                    due = float(created) + horizon * 60
                    if now < due:
                        continue
                    with evidence_db() as db:
                        exists = db.execute(
                            "SELECT 1 FROM signal_outcomes WHERE signal_uid=? AND horizon_min=?",
                            (uid, horizon)
                        ).fetchone()
                    if exists:
                        continue
                    ts = pd.to_datetime(float(created), unit="s")
                    if getattr(d.index, "tz", None) is not None:
                        ts = ts.tz_localize(d.index.tz)
                    # Use the first candle at/after the signal timestamp.
                    pos = d.index.searchsorted(ts)
                    if pos >= len(d):
                        continue
                    metrics = _evidence_path_metrics(
                        d, int(pos), _discovery_horizon_bars(int(tf), horizon), direction
                    )
                    if not metrics:
                        continue
                    with evidence_db() as db:
                        db.execute("""
                            INSERT OR REPLACE INTO signal_outcomes(
                                signal_uid,horizon_min,evaluated_at,future_price,
                                signed_return_pct,mfe_pct,mae_pct,threshold_pct,
                                result,bars_observed
                            ) VALUES(?,?,?,?,?,?,?,?,?,?)
                        """, (
                            uid, horizon, metrics["evaluated_at"], metrics["future_price"],
                            metrics["signed_return_pct"], metrics["mfe_pct"],
                            metrics["mae_pct"], metrics["threshold_pct"],
                            metrics["result"], metrics["bars_observed"]
                        ))
                    done += 1

                with evidence_db() as db:
                    count = db.execute(
                        "SELECT COUNT(*) FROM signal_outcomes WHERE signal_uid=?",
                        (uid,)
                    ).fetchone()[0]
                    if count >= len(EVIDENCE_HORIZONS):
                        db.execute(
                            "UPDATE signal_ledger SET status='COMPLETE' WHERE signal_uid=?",
                            (uid,)
                        )
            except Exception as exc:
                logging.debug("evidence validation %s failed: %s", uid, exc)
    except Exception as exc:
        logging.debug("evidence_validate_pending failed: %s", exc)
    return done


def evidence_validator_worker():
    global _EVIDENCE_LAST_SCAN
    while True:
        try:
            n = evidence_validate_pending()
            _EVIDENCE_LAST_SCAN = time.time()
            if n:
                evidence_rebuild_stats()
        except Exception:
            pass
        time.sleep(60)


def evidence_start_validator():
    global _EVIDENCE_VALIDATOR_STARTED
    with _EVIDENCE_LOCK:
        if _EVIDENCE_VALIDATOR_STARTED:
            return
        _EVIDENCE_VALIDATOR_STARTED = True
    threading.Thread(
        target=evidence_validator_worker,
        daemon=True,
        name="EvidenceSignalValidator"
    ).start()


def _evidence_aggregate(rows):
    rows = list(rows)
    n = len(rows)
    wins = sum(1 for r in rows if r[0] == "WIN")
    losses = sum(1 for r in rows if r[0] in ("LOSS", "FAKE"))
    fake = sum(1 for r in rows if r[0] == "FAKE")
    neutral = sum(1 for r in rows if r[0] == "NEUTRAL")
    rets = [_evidence_float(r[1]) for r in rows]
    mfe = [_evidence_float(r[2]) for r in rows]
    mae = [_evidence_float(r[3]) for r in rows]
    gross_profit = sum(x for x in rets if x > 0)
    gross_loss = abs(sum(x for x in rets if x < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    win_rate = wins / n * 100 if n else 0.0
    precision = wins / max(1, wins + losses) * 100
    avg_ret = sum(rets) / n if n else 0.0
    expectancy = avg_ret
    return {
        "samples": n, "wins": wins, "losses": losses, "fake": fake,
        "neutral": neutral, "win_rate": win_rate, "precision": precision,
        "avg_return": avg_ret, "expectancy": expectancy, "profit_factor": pf,
        "avg_mfe": sum(mfe) / len(mfe) if mfe else 0.0,
        "avg_mae": sum(mae) / len(mae) if mae else 0.0,
    }


def evidence_rebuild_stats():
    try:
        with evidence_db() as db:
            db.execute("DELETE FROM rule_stats")
            db.execute("DELETE FROM combo_stats")
            rows = db.execute("""
                SELECT h.result,h.signed_return_pct,h.mfe_pct,h.mae_pct,
                       l.rules_json,l.direction,h.horizon_min
                FROM signal_outcomes h
                JOIN signal_ledger l ON l.signal_uid=h.signal_uid
                WHERE h.result IS NOT NULL
            """).fetchall()

            buckets = defaultdict(list)
            combos = defaultdict(list)
            for result, ret, mfe, mae, rules_json, direction, horizon in rows:
                try:
                    rules = json.loads(rules_json or "[]")
                except Exception:
                    rules = []
                payload = (result, ret, mfe, mae)
                for rule in rules:
                    buckets[(str(rule), direction, int(horizon))].append(payload)
                if rules:
                    # Limit combinations to size 2-3 to avoid sparse combinatorial noise.
                    unique = sorted(set(map(str, rules)))
                    for k in (2, 3):
                        if len(unique) >= k:
                            for combo in _evidence_itertools.combinations(unique, k):
                                combos[("|".join(combo), direction, int(horizon))].append(payload)

            now = time.time()
            for key, vals in buckets.items():
                rule, direction, horizon = key
                a = _evidence_aggregate(vals)
                db.execute("""
                    INSERT INTO rule_stats(
                        rule_name,direction,horizon_min,samples,wins,losses,fake,
                        neutral,win_rate,precision_val,avg_return,expectancy,
                        profit_factor,avg_mfe,avg_mae,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    rule,direction,horizon,a["samples"],a["wins"],a["losses"],a["fake"],
                    a["neutral"],a["win_rate"],a["precision"],a["avg_return"],
                    a["expectancy"],a["profit_factor"],a["avg_mfe"],a["avg_mae"],now
                ))
            for key, vals in combos.items():
                combo, direction, horizon = key
                a = _evidence_aggregate(vals)
                db.execute("""
                    INSERT INTO combo_stats(
                        combo_key,direction,horizon_min,samples,wins,losses,fake,
                        neutral,win_rate,precision_val,avg_return,expectancy,
                        profit_factor,avg_mfe,avg_mae,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    combo,direction,horizon,a["samples"],a["wins"],a["losses"],a["fake"],
                    a["neutral"],a["win_rate"],a["precision"],a["avg_return"],
                    a["expectancy"],a["profit_factor"],a["avg_mfe"],a["avg_mae"],now
                ))
    except Exception as exc:
        logging.debug("evidence_rebuild_stats failed: %s", exc)


def _evidence_historical_signal_at(df, i):
    """Reconstruct a historical SMC setup using only information <= i.

    Improvements over ESL5:
    - bounded local structure window (prevents quadratic growth on long history)
    - explicit BOS/CHoCH + liquidity + OB/FVG confluence
    - higher-timeframe trend confirmation built only from candles already closed
    - no future-dependent rule or score
    """
    try:
        if df is None or i < 180 or i >= len(df):
            return None
        w = df.iloc[max(0, i - 1200):i + 1].copy()
        c = _evidence_float(w["close"].iloc[-1])
        if c <= 0 or len(w) < 180:
            return None

        swings_h, swings_l = detect_swing_points(w, left=2, right=2)
        if len(swings_h) < 3 or len(swings_l) < 3:
            return None
        bos = detect_bos_choch(w, swings_h, swings_l)
        sweep_high, sweep_low = detect_liquidity_sweeps(w, swings_h, swings_l)
        bull_obs, bear_obs = detect_order_blocks(w, swings_h, swings_l)
        bull_fvg, bear_fvg = detect_fvg(w)
        imbalance_score = 0.0
        try:
            imbalance_score, _, _ = detect_market_imbalance_score(w)
        except Exception:
            pass

        rsi = _evidence_float(compute_rsi_series(w, 14).iloc[-1], 50)
        adx = _evidence_float(calculate_adx(w, 14), 0)
        ema9 = _evidence_float(w["close"].ewm(span=9, adjust=False).mean().iloc[-1])
        ema21 = _evidence_float(w["close"].ewm(span=21, adjust=False).mean().iloc[-1])
        ema50 = _evidence_float(w["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        ema100 = _evidence_float(w["close"].ewm(span=100, adjust=False).mean().iloc[-1])
        macd_line = w["close"].ewm(span=12, adjust=False).mean() - w["close"].ewm(span=26, adjust=False).mean()
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = _evidence_float(macd_line.iloc[-1] - macd_signal.iloc[-1])
        vol_ma = _evidence_float(w["volume"].rolling(48).mean().iloc[-1], 0)
        vol_ratio = _evidence_float(w["volume"].iloc[-1] / max(vol_ma, 1e-12), 1)
        vpin = _evidence_float(estimate_vpin(w), 0)
        poc, _, _ = calculate_volume_profile(w.tail(160), bins=24)

        # Higher-timeframe confirmation from already-closed bars only.
        mtf = {}
        for mult, name in ((4, "HTF4"), (16, "HTF16")):
            x = w.iloc[::mult].copy()
            if len(x) >= 40:
                e20 = _evidence_float(x["close"].ewm(span=20, adjust=False).mean().iloc[-1])
                e50 = _evidence_float(x["close"].ewm(span=50, adjust=False).mean().iloc[-1])
                mtf[name] = 1 if e20 > e50 else -1 if e20 < e50 else 0
            else:
                mtf[name] = 0
        mtf_align = sum(mtf.values()) / max(1, len(mtf))

        score = 0.0
        rules, reasons = [], []
        if bos.get("bos_bullish"):
            score += 3.2; rules.append("BOS-Bullish"); reasons.append("BOS صعودی")
        if bos.get("bos_bearish"):
            score -= 3.2; rules.append("BOS-Bearish"); reasons.append("BOS نزولی")
        if bos.get("choch_bullish"):
            score += 3.6; rules.append("CHoCH-Bullish"); reasons.append("CHoCH صعودی")
        if bos.get("choch_bearish"):
            score -= 3.6; rules.append("CHoCH-Bearish"); reasons.append("CHoCH نزولی")
        if sweep_low:
            score += 2.6; rules.append("Liquidity-Sweep-Low"); reasons.append("جمع‌آوری نقدینگی پایین")
        if sweep_high:
            score -= 2.6; rules.append("Liquidity-Sweep-High"); reasons.append("جمع‌آوری نقدینگی بالا")

        if bull_fvg and any(int(x.get("index", -999)) >= len(w) - 10 for x in bull_fvg):
            score += 1.8; rules.append("FVG-Bullish"); reasons.append("FVG صعودی اخیر")
        if bear_fvg and any(int(x.get("index", -999)) >= len(w) - 10 for x in bear_fvg):
            score -= 1.8; rules.append("FVG-Bearish"); reasons.append("FVG نزولی اخیر")
        if bull_obs:
            ob = bull_obs[-1]
            if float(ob.get("low", c)) <= c <= float(ob.get("high", c)) * 1.012:
                score += 2.2; rules.append("OrderBlock-Bullish"); reasons.append("حمایت Order Block")
        if bear_obs:
            ob = bear_obs[-1]
            if float(ob.get("low", c)) * 0.988 <= c <= float(ob.get("high", c)):
                score -= 2.2; rules.append("OrderBlock-Bearish"); reasons.append("مقاومت Order Block")

        if ema9 > ema21 > ema50 and c > ema100:
            score += 1.8; rules.append("EMA-Alignment-Bull")
        elif ema9 < ema21 < ema50 and c < ema100:
            score -= 1.8; rules.append("EMA-Alignment-Bear")
        if macd_hist > 0:
            score += 0.8; rules.append("MACD-Hist-Bull")
        elif macd_hist < 0:
            score -= 0.8; rules.append("MACD-Hist-Bear")
        if rsi >= 54:
            score += 0.6; rules.append("RSI-Momentum-Bull")
        elif rsi <= 46:
            score -= 0.6; rules.append("RSI-Momentum-Bear")
        if adx >= 22:
            rules.append("ADX-Trend")
        if vol_ratio >= 1.30:
            if score > 0:
                score += 1.2; rules.append("Volume-Expansion"); reasons.append("افزایش حجم")
            elif score < 0:
                score -= 1.2; rules.append("Volume-Expansion"); reasons.append("افزایش حجم")
        if imbalance_score > 0.10:
            score += 0.7; rules.append("Imbalance-Bullish")
        elif imbalance_score < -0.10:
            score -= 0.7; rules.append("Imbalance-Bearish")
        if mtf_align >= 0.5 and score > 0:
            score += 1.4; rules.append("HTF-Alignment-Bull")
        elif mtf_align <= -0.5 and score < 0:
            score -= 1.4; rules.append("HTF-Alignment-Bear")
        if poc is not None:
            if c > poc and score > 0:
                score += 0.4; rules.append("Above-POC")
            elif c < poc and score < 0:
                score -= 0.4; rules.append("Below-POC")
        if vpin >= 0.70:
            reasons.append("VPIN بالا؛ ریسک مسیر بیشتر")

        direction = "LONG" if score > 0 else "SHORT" if score < 0 else "WAIT"
        unique_rules = list(dict.fromkeys(rules))
        smc_rules = {"BOS-Bullish","BOS-Bearish","CHoCH-Bullish","CHoCH-Bearish",
                     "Liquidity-Sweep-Low","Liquidity-Sweep-High","FVG-Bullish",
                     "FVG-Bearish","OrderBlock-Bullish","OrderBlock-Bearish"}
        smc_count = len(set(unique_rules) & smc_rules)
        # Require both structure and confirmation. This is intentionally stricter
        # than ESL5 to reduce the number of noisy historical samples.
        if direction == "WAIT" or abs(score) < 6.0 or len(unique_rules) < 3 or smc_count < 1:
            return None
        quality = float(np.clip(50 + abs(score) * 5.5 + (8 if adx >= 22 else 0)
                                + 7 * max(0, abs(mtf_align)), 0, 100))
        return {
            "direction": direction, "entry": c, "score": float(score), "quality": quality,
            "rsi": rsi, "adx": adx, "macd_hist": macd_hist, "vol_ratio": vol_ratio,
            "vpin": vpin, "imbalance_score": imbalance_score, "poc": poc,
            "mtf_align": mtf_align, "mtf": mtf, "bos": bos,
            "sweep_high": bool(sweep_high), "sweep_low": bool(sweep_low),
            "rules": unique_rules, "reasons": reasons,
        }
    except Exception as exc:
        logging.debug("historical SMC signal reconstruction failed: %s", exc)
        return None

def evidence_historical_mine(symbol, timeframe=15, bars=5000, run_id=None):
    """Historical SMC signal mining with chronological, leak-free labels.
    Signal creation uses only candles available at the signal timestamp;
    future candles are used solely for WIN/LOSS/FAKE/NEUTRAL validation.
    """
    key = canonical(symbol)
    tf = int(timeframe)
    run_id = run_id or f"HM-SMC-{key}-{tf}-{int(time.time())}"
    df = _discovery_get_history(key, tf, bars=min(8000, int(bars)))
    if df is None or df.empty or len(df) < 500:
        return {"run_id": run_id, "signals": 0, "error": "History ناکافی"}

    with evidence_db() as db:
        db.execute("""
            INSERT OR REPLACE INTO research_runs(
                run_id,started_at,status,symbols,timeframes,signals,message
            ) VALUES(?,?,?,?,?,?,?)
        """, (run_id, time.time(), "running", 1, 1, 0,
              f"SMC historical mining {key}/{tf} | leak-free path labels + HTF confirmation"))

    signals = 0
    # Reserve enough future candles for the 12h label.
    max_hb = _discovery_horizon_bars(tf, max(EVIDENCE_HORIZONS))
    start = 120
    stop = len(df) - max_hb - 1

    for i in range(start, max(start, stop)):
        sig = _evidence_historical_signal_at(df, i)
        if not sig:
            continue
        ts = float(pd.Timestamp(df.index[i]).timestamp())
        signals += 1
        for horizon in EVIDENCE_HORIZONS:
            m = _evidence_path_metrics(
                df, i, _discovery_horizon_bars(tf, horizon), sig["direction"]
            )
            if not m:
                continue
            features = {k: v for k, v in sig.items()
                        if k not in {"rules", "reasons", "direction", "entry", "score"}}
            with evidence_db() as db:
                db.execute("""
                    INSERT INTO historical_signals(
                        run_id,symbol,timeframe,ts,direction,entry,score,
                        rules_json,features_json,horizon_min,future_price,
                        signed_return_pct,mfe_pct,mae_pct,threshold_pct,result,
                        target_hit,stop_hit,ambiguous_path,bars_to_target,bars_to_stop,
                        bars_to_mfe,bars_to_mae,label_confidence
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    run_id, key, tf, ts, sig["direction"], sig["entry"], sig["score"],
                    _evidence_json(sig["rules"]), _evidence_json(features), horizon,
                    m["future_price"], m["signed_return_pct"], m["mfe_pct"],
                    m["mae_pct"], m["threshold_pct"], m["result"],
                    m.get("target_hit",0), m.get("stop_hit",0), m.get("ambiguous_path",0),
                    m.get("bars_to_target"), m.get("bars_to_stop"), m.get("bars_to_mfe"),
                    m.get("bars_to_mae"), m.get("label_confidence",1.0)
                ))

    with evidence_db() as db:
        db.execute("""
            UPDATE research_runs
            SET finished_at=?,status='done',signals=?,message=?
            WHERE run_id=?
        """, (time.time(), signals,
              f"SMC mining completed: {key}/{tf} | {signals} signals | path-aware labels", run_id))

    # Build cumulative historical statistics, not just statistics for this run.
    evidence_rebuild_historical_stats()
    return {"run_id": run_id, "signals": signals, "symbol": key,
            "timeframe": tf, "engine": EVIDENCE_VERSION, "mode": "SMC"}


def evidence_rebuild_historical_stats(run_id=None):
    """
    Aggregate historical mining independently from live signal outcomes.
    Uses the same result semantics, so before/after comparisons are meaningful.
    """
    try:
        with evidence_db() as db:
            where = "WHERE 1=1"
            args = []
            if run_id:
                where += " AND run_id=?"
                args.append(run_id)
            rows = db.execute(f"""
                SELECT result,signed_return_pct,mfe_pct,mae_pct,
                       rules_json,direction,horizon_min
                FROM historical_signals {where}
            """, args).fetchall()

            # Store historical aggregates in the same tables by rebuilding
            # from all historical data. This makes the displayed stats cumulative.
            db.execute("DELETE FROM rule_stats")
            db.execute("DELETE FROM combo_stats")

            buckets = defaultdict(list)
            combos = defaultdict(list)
            for result, ret, mfe, mae, rules_json, direction, horizon in rows:
                try:
                    rules = json.loads(rules_json or "[]")
                except Exception:
                    rules = []
                payload = (result, ret, mfe, mae)
                unique = sorted(set(map(str, rules)))
                for rule in unique:
                    buckets[(rule,direction,int(horizon))].append(payload)
                for k in (2,3):
                    if len(unique) >= k:
                        for combo in _evidence_itertools.combinations(unique,k):
                            combos[("|".join(combo),direction,int(horizon))].append(payload)

            now = time.time()
            for (rule,direction,horizon), vals in buckets.items():
                a = _evidence_aggregate(vals)
                if a["samples"] < 1:
                    continue
                db.execute("""
                    INSERT INTO rule_stats VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    rule,direction,horizon,a["samples"],a["wins"],a["losses"],a["fake"],
                    a["neutral"],a["win_rate"],a["precision"],a["avg_return"],
                    a["expectancy"],a["profit_factor"],a["avg_mfe"],a["avg_mae"],now
                ))
            for (combo,direction,horizon), vals in combos.items():
                a = _evidence_aggregate(vals)
                db.execute("""
                    INSERT INTO combo_stats VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    combo,direction,horizon,a["samples"],a["wins"],a["losses"],a["fake"],
                    a["neutral"],a["win_rate"],a["precision"],a["avg_return"],
                    a["expectancy"],a["profit_factor"],a["avg_mfe"],a["avg_mae"],now
                ))
    except Exception as exc:
        logging.debug("evidence_rebuild_historical_stats failed: %s", exc)


def evidence_compare_engines(before_metrics=None, after_metrics=None):
    """
    Small objective comparison helper. Returns only measurable deltas.
    """
    def norm(x):
        x = dict(x or {})
        return {
            "win_rate": _evidence_float(x.get("win_rate")),
            "precision": _evidence_float(x.get("precision")),
            "avg_return": _evidence_float(x.get("avg_return")),
            "expectancy": _evidence_float(x.get("expectancy")),
            "profit_factor": _evidence_float(x.get("profit_factor")),
            "fake_rate": _evidence_float(x.get("fake_rate")),
        }
    b, a = norm(before_metrics), norm(after_metrics)
    return {k: a[k] - b[k] for k in a}


def evidence_dashboard_data(limit=30):
    with evidence_db() as db:
        top_rules = db.execute("""
            SELECT rule_name,direction,horizon_min,samples,wins,losses,fake,
                   neutral,win_rate,precision_val,avg_return,expectancy,
                   profit_factor,avg_mfe,avg_mae
            FROM rule_stats
            WHERE samples >= ?
            ORDER BY expectancy DESC, precision_val DESC
            LIMIT ?
        """, (EVIDENCE_MIN_RULE_SAMPLES, int(limit))).fetchall()
        top_combos = db.execute("""
            SELECT combo_key,direction,horizon_min,samples,wins,losses,fake,
                   neutral,win_rate,precision_val,avg_return,expectancy,
                   profit_factor,avg_mfe,avg_mae
            FROM combo_stats
            WHERE samples >= ?
            ORDER BY expectancy DESC, precision_val DESC
            LIMIT ?
        """, (EVIDENCE_MIN_RULE_SAMPLES, int(limit))).fetchall()
        overall = db.execute("""
            SELECT h.result,h.signed_return_pct,h.mfe_pct,h.mae_pct
            FROM signal_outcomes h
        """).fetchall()
    return {
        "top_rules": top_rules,
        "top_combos": top_combos,
        "overall": _evidence_aggregate(overall)
    }


def evidence_format_dashboard():
    d = evidence_dashboard_data(25)
    a = d["overall"]
    lines = [
        "=" * 100,
        f"Evidence Signal Ledger {EVIDENCE_VERSION}",
        "=" * 100,
        f"Live outcomes | Samples={a['samples']} | WIN={a['wins']} | LOSS={a['losses']} | "
        f"FAKE={a['fake']} | NEUTRAL={a['neutral']}",
        f"Win Rate={a['win_rate']:.2f}% | Precision={a['precision']:.2f}% | "
        f"Avg Return={a['avg_return']:.3f}% | PF={a['profit_factor']:.2f}",
        f"Avg MFE={a['avg_mfe']:.3f}% | Avg MAE={a['avg_mae']:.3f}%",
        "",
        "=== بهترین Ruleها ==="
    ]
    for r in d["top_rules"][:15]:
        lines.append(
            f"{r[0]} | {r[1]} | {r[2]}m | n={r[3]} | WIN={r[4]} | LOSS={r[5]} | "
            f"FAKE={r[6]} | WR={r[8]:.1f}% | Precision={r[9]:.1f}% | "
            f"Exp={r[11]:+.3f}% | PF={r[12]:.2f}"
        )
    lines += ["", "=== بهترین ترکیب Ruleها ==="]
    for r in d["top_combos"][:15]:
        lines.append(
            f"{r[0]} | {r[1]} | {r[2]}m | n={r[3]} | WR={r[8]:.1f}% | "
            f"Precision={r[9]:.1f}% | Exp={r[11]:+.3f}% | PF={r[12]:.2f}"
        )
    return "\n".join(lines)


# ---- Integrate with the final unified decision without replacing its logic. ----
try:
    _unified_ai_decision_before_evidence = unified_ai_decision
    def unified_ai_decision(sym, timeframe=15):
        result = _unified_ai_decision_before_evidence(sym, timeframe)
        try:
            uid = evidence_record_signal(result, timeframe)
            if uid:
                result["evidence_signal_uid"] = uid
                result["evidence_engine_version"] = EVIDENCE_VERSION
        except Exception as exc:
            logging.debug("evidence live hook failed: %s", exc)
        return result
except Exception:
    pass


# ---- Add a dedicated GUI tab. Existing tabs and engines remain intact. ----
def _evidence_build_tab(self):
    try:
        tab = _make_scrollable_tab(self.notebook, bg="#0f172a")
        self.notebook.add(tab._scroll_outer, text="🟡 ارزیابی سیگنال")

        top = tk.Frame(tab, bg="#1e293b")
        top.pack(fill="x", padx=8, pady=8)

        tk.Label(
            top,
            text="Signal Evidence Ledger — ثبت، اعتبارسنجی، MFE/MAE و Rule Analytics",
            bg="#1e293b", fg="white", font=("Tahoma", 12, "bold")
        ).pack(side="left", padx=8)

        def btn(text, cmd):
            return tk.Button(top, text=text, command=cmd, bg=CTRL_BTN_BG, fg="white")

        btn("اعتبارسنجی زنده", self._evidence_validate_now).pack(side="left", padx=4)
        btn("استخراج تاریخی", self._evidence_mine_dialog).pack(side="left", padx=4)
        btn("بازسازی آمار", self._evidence_refresh).pack(side="left", padx=4)
        btn("Export CSV", self._evidence_export).pack(side="left", padx=4)

        self.evidence_status = tk.Label(
            tab, text="آماده", anchor="w", bg="#0b1220", fg="#e5e7eb"
        )
        self.evidence_status.pack(fill="x", padx=8, pady=(0, 5))

        cols = (
            "rule","direction","tf","samples","wins","losses","fake",
            "neutral","winrate","precision","expectancy","pf","mfe","mae"
        )
        frame = tk.Frame(tab)
        frame.pack(fill="both", expand=True, padx=8, pady=5)
        self.evidence_tree = ttk.Treeview(frame, columns=cols, show="headings", height=14)
        headings = {
            "rule":"Rule / Combination","direction":"Direction","tf":"Horizon",
            "samples":"N","wins":"WIN","losses":"LOSS","fake":"FAKE","neutral":"NEUTRAL",
            "winrate":"Win Rate","precision":"Precision","expectancy":"Expectancy",
            "pf":"PF","mfe":"MFE","mae":"MAE"
        }
        for c in cols:
            self.evidence_tree.heading(c, text=headings[c])
            self.evidence_tree.column(c, width=105 if c != "rule" else 320, anchor="center")
        ysb = ttk.Scrollbar(frame, orient="vertical", command=self.evidence_tree.yview)
        self.evidence_tree.configure(yscrollcommand=ysb.set)
        self.evidence_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")

        self.evidence_text = tk.Text(
            tab, height=15, bg="#0b1220", fg="#e5e7eb",
            font=("Consolas", 10), wrap="none"
        )
        self.evidence_text.pack(fill="both", expand=True, padx=8, pady=5)

        self._evidence_refresh()
    except Exception as exc:
        logging.debug("build evidence tab failed: %s", exc)


def _evidence_refresh(self):
    try:
        evidence_rebuild_stats()
        data = evidence_dashboard_data(50)
        self.evidence_tree.delete(*self.evidence_tree.get_children())
        for r in data["top_rules"]:
            vals = (
                r[0], r[1], f"{r[2]}m", r[3], r[4], r[5], r[6], r[7],
                f"{r[8]:.1f}%", f"{r[9]:.1f}%", f"{r[11]:+.3f}%",
                "∞" if r[12] == float("inf") else f"{r[12]:.2f}",
                f"{r[13]:+.3f}%", f"{r[14]:+.3f}%"
            )
            self.evidence_tree.insert("", "end", values=vals)
        self.evidence_text.delete("1.0", "end")
        self.evidence_text.insert("1.0", evidence_format_dashboard())
        self.evidence_status.config(
            text=f"آخرین بروزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | DB: {EVIDENCE_DB}"
        )
    except Exception as exc:
        logging.debug("evidence refresh failed: %s", exc)


def _evidence_validate_now(self):
    def worker():
        try:
            n = evidence_validate_pending(1000)
            evidence_rebuild_stats()
            self.root.after(0, lambda: (
                self._evidence_refresh(),
                self.evidence_status.config(text=f"اعتبارسنجی انجام شد | {n} outcome جدید")
            ))
        except Exception as exc:
            self.root.after(0, lambda e=str(exc): self.evidence_status.config(text=f"خطا: {e}"))
    threading.Thread(target=worker, daemon=True, name="EvidenceManualValidate").start()


def _evidence_mine_dialog(self):
    try:
        symbol = self.structure_symbol_combo.get().strip() if hasattr(self, "structure_symbol_combo") else ""
        if not symbol:
            symbol = SYMBOLS[0] if SYMBOLS else ""
        symbol = simpledialog.askstring("Historical Signal Mining", "نماد:", initialvalue=symbol, parent=self.root)
        if not symbol:
            return
        tf = simpledialog.askinteger("Historical Signal Mining", "تایم‌فریم (دقیقه):", initialvalue=15, minvalue=1, maxvalue=1440, parent=self.root)
        if not tf:
            return
        bars = simpledialog.askinteger("Historical Signal Mining", "تعداد کندل:", initialvalue=5000, minvalue=500, maxvalue=8000, parent=self.root)
        if not bars:
            return

        self.evidence_status.config(text=f"در حال استخراج تاریخی {canonical(symbol)} / {tf}m ...")

        def worker():
            try:
                res = evidence_historical_mine(symbol, tf, bars)
                msg = f"پایان | {res.get('symbol')} {res.get('timeframe')}m | Signals={res.get('signals',0)}"
                self.root.after(0, lambda: (
                    self._evidence_refresh(),
                    self.evidence_status.config(text=msg)
                ))
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): self.evidence_status.config(text=f"خطا: {e}"))

        threading.Thread(target=worker, daemon=True, name="HistoricalSignalMining").start()
    except Exception as exc:
        logging.debug("evidence mine dialog failed: %s", exc)


def _evidence_export(self):
    try:
        p = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")]
        )
        if not p:
            return
        with evidence_db() as db, open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([
                "signal_uid","symbol","timeframe","direction","decision","created_at",
                "entry","score","confidence","prob_long","prob_short","quality",
                "rules","horizon_min","future_price","signed_return_pct",
                "mfe_pct","mae_pct","threshold_pct","result"
            ])
            rows = db.execute("""
                SELECT l.signal_uid,l.symbol,l.timeframe,l.direction,l.decision,l.created_at,
                       l.entry,l.score,l.confidence,l.prob_long,l.prob_short,l.quality,
                       l.rules_json,o.horizon_min,o.future_price,o.signed_return_pct,
                       o.mfe_pct,o.mae_pct,o.threshold_pct,o.result
                FROM signal_ledger l
                LEFT JOIN signal_outcomes o ON o.signal_uid=l.signal_uid
                ORDER BY l.created_at DESC
            """).fetchall()
            for r in rows:
                w.writerow(r)
        self.evidence_status.config(text=f"Export شد: {p}")
    except Exception as exc:
        messagebox.showerror("Evidence Export", str(exc))


# Bind methods once.
NobitexScalperPro._evidence_build_tab = _evidence_build_tab
NobitexScalperPro._evidence_refresh = _evidence_refresh
NobitexScalperPro._evidence_validate_now = _evidence_validate_now
NobitexScalperPro._evidence_mine_dialog = _evidence_mine_dialog
NobitexScalperPro._evidence_export = _evidence_export

try:
    _NobitexScalperPro_init_original_evidence = NobitexScalperPro.__init__
    def _NobitexScalperPro_init_evidence(self, root):
        _NobitexScalperPro_init_original_evidence(self, root)
        try:
            self._evidence_build_tab()
            evidence_start_validator()
        except Exception as exc:
            logging.debug("evidence integration init failed: %s", exc)
    NobitexScalperPro.__init__ = _NobitexScalperPro_init_evidence
except Exception:
    pass





# ============================================================
# UNIFIED MARKET INTELLIGENCE CENTER V1
# ============================================================
# یک هسته واحد برای:
# تاریخچه -> ساختار بازار -> ریسک -> بک تست/OOS -> بهینه سازی -> ML ->
# کالیبراسیون -> پیش بینی حرکت -> ذخیره دائمی -> بررسی تازگی در شروع برنامه.
# The formerly separate research/learning/market/risk/backtest tabs are integrated here.
# Their standalone Notebook tabs are intentionally not added; other independent tabs remain.

UNIFIED_TIMEFRAMES = (1440, 240, 60, 30, 15, 5, 1)
UNIFIED_SCHEMA = 2
UNIFIED_PIPELINE_GENERATION = 11
DATA_PIPELINE_FIX_VERSION = "V7_HIGH_TO_LOW_FULL_AUDIT"
UNIFIED_PIPELINE_VERSION = "V12_DASHBOARD_GENERATION_FIX"
UNIFIED_DEFAULT_YEARS = 2
UNIFIED_STALE_HOURS = {1: 6, 5: 6, 15: 12, 30: 12, 60: 24, 240: 36, 1440: 72}
UNIFIED_CONFIG_FILE = Path(APP_DATA_DIR) / "unified_center_config.json"
UNIFIED_DEFAULT_DIR = Path(APP_DATA_DIR) / "unified_learning"
UNIFIED_STORAGE_DIR = UNIFIED_DEFAULT_DIR
UNIFIED_DB_FILE = UNIFIED_STORAGE_DIR / "NobitexUnifiedLearning.sqlite3"
UNIFIED_RUN_FILE = UNIFIED_STORAGE_DIR / "last_unified_run.json"
UNIFIED_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _unified_load_config():
    global UNIFIED_STORAGE_DIR, UNIFIED_DB_FILE, UNIFIED_RUN_FILE
    try:
        if UNIFIED_CONFIG_FILE.exists():
            d=json.loads(UNIFIED_CONFIG_FILE.read_text(encoding="utf-8"))
            folder=str(d.get("storage_dir") or "").strip()
            if folder:
                UNIFIED_STORAGE_DIR=Path(folder).expanduser().resolve()
    except Exception as exc:
        logging.debug("unified config load: %s", exc)
    UNIFIED_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    UNIFIED_DB_FILE=UNIFIED_STORAGE_DIR / "NobitexUnifiedLearning.sqlite3"
    UNIFIED_RUN_FILE=UNIFIED_STORAGE_DIR / "last_unified_run.json"


_unified_load_config()


def _unified_db():
    conn=sqlite3.connect(str(UNIFIED_DB_FILE), timeout=60, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _unified_init_db():
    UNIFIED_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with _unified_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS training_state(
            symbol TEXT NOT NULL, timeframe INTEGER NOT NULL,
            history_years REAL NOT NULL DEFAULT 2,
            history_last INTEGER, history_count INTEGER DEFAULT 0,
            history_expected INTEGER DEFAULT 0, coverage REAL DEFAULT 0,
            model_trained_at INTEGER DEFAULT 0, discovery_trained_at INTEGER DEFAULT 0,
            optimizer_at INTEGER DEFAULT 0, deep_research_at INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending', message TEXT DEFAULT '',
            last_error TEXT DEFAULT '', version TEXT DEFAULT '',
            pipeline_generation INTEGER DEFAULT 0,
            PRIMARY KEY(symbol,timeframe)
        );
        CREATE TABLE IF NOT EXISTS predictions(
            symbol TEXT NOT NULL, timeframe INTEGER NOT NULL,
            ts INTEGER NOT NULL, price REAL, direction TEXT,
            prob_long REAL, prob_short REAL, prob_neutral REAL,
            expected_move REAL, confidence REAL, quality REAL,
            decision TEXT, model TEXT, trained_at INTEGER,
            PRIMARY KEY(symbol,timeframe)
        );
        CREATE TABLE IF NOT EXISTS runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT, started_at INTEGER,
            finished_at INTEGER, status TEXT, total INTEGER,
            completed INTEGER, updated INTEGER, skipped INTEGER, failed INTEGER,
            message TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_predictions_rank ON predictions(expected_move DESC);
        """)
        db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema',?)",(str(UNIFIED_SCHEMA),))
        db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('storage_dir',?)",(str(UNIFIED_STORAGE_DIR),))
        try:
            cols={str(r[1]) for r in db.execute("PRAGMA table_info(training_state)").fetchall()}
            if 'pipeline_generation' not in cols:
                db.execute("ALTER TABLE training_state ADD COLUMN pipeline_generation INTEGER DEFAULT 0")
            # Existing models belong to older pipeline builds. Keep their candles and
            # history, but do not count their model as ready until this build retrains it.
            db.execute("UPDATE training_state SET pipeline_generation=0 WHERE pipeline_generation IS NULL")
        except Exception:
            pass



_unified_init_db()


def _unified_set_storage(folder):
    global UNIFIED_STORAGE_DIR, UNIFIED_DB_FILE, UNIFIED_RUN_FILE
    folder=str(folder or "").strip()
    if not folder:
        return False
    path=Path(folder).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    UNIFIED_STORAGE_DIR=path
    UNIFIED_DB_FILE=path / "NobitexUnifiedLearning.sqlite3"
    UNIFIED_RUN_FILE=path / "last_unified_run.json"
    UNIFIED_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    UNIFIED_CONFIG_FILE.write_text(json.dumps({"storage_dir":str(path)},ensure_ascii=False,indent=2),encoding="utf-8")
    # Existing engines use these globals. Point them to the same persistent root
    # so history/research/model indexes share one operator-selected location.
    global ADAPTIVE_DB_FILE, DISCOVERY_DIR, DEEP_RESEARCH_DIR, DEEP_RESEARCH_BACKUP_DIR, DEEP_RESEARCH_MASTER_FILE
    ADAPTIVE_DB_FILE=str(path / "NobitexUnifiedLearning.sqlite3")
    DISCOVERY_DIR=path / "models"
    DISCOVERY_DIR.mkdir(parents=True,exist_ok=True)
    DEEP_RESEARCH_DIR=path / "deep_research"
    DEEP_RESEARCH_DIR.mkdir(parents=True,exist_ok=True)
    DEEP_RESEARCH_BACKUP_DIR=DEEP_RESEARCH_DIR / "backups"
    DEEP_RESEARCH_BACKUP_DIR.mkdir(parents=True,exist_ok=True)
    DEEP_RESEARCH_MASTER_FILE=DEEP_RESEARCH_DIR / "research_master.json"
    try:
        globals()['_adaptive_initialized']=False
        adaptive_init_db()
    except Exception as exc:
        logging.debug("unified storage adaptive init: %s",exc)
    _unified_init_db()
    return True


def _unified_symbols():
    syms=[canonical(x) for x in list(SYMBOLS or []) if canonical(x)]
    seen=set(); out=[]
    for x in syms:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def _unified_state(symbol,tf):
    try:
        _unified_init_db()
        with _unified_db() as db:
            r=db.execute("SELECT symbol,timeframe,history_years,history_last,history_count,history_expected,coverage,model_trained_at,discovery_trained_at,optimizer_at,deep_research_at,status,message,last_error,version,pipeline_generation FROM training_state WHERE symbol=? AND timeframe=?",(canonical(symbol),int(tf))).fetchone()
        if not r: return None
        keys=['symbol','timeframe','history_years','history_last','history_count','history_expected','coverage','model_trained_at','discovery_trained_at','optimizer_at','deep_research_at','status','message','last_error','version','pipeline_generation']
        return dict(zip(keys,r))
    except Exception:
        return None


def _unified_upsert_state(symbol,tf,**kwargs):
    key=canonical(symbol); tf=int(tf); old=_unified_state(key,tf) or {}
    old.update(kwargs); old.update(symbol=key,timeframe=tf)
    fields=['history_years','history_last','history_count','history_expected','coverage','model_trained_at','discovery_trained_at','optimizer_at','deep_research_at','status','message','last_error','version','pipeline_generation']
    with _unified_db() as db:
        db.execute("""INSERT OR REPLACE INTO training_state
        (symbol,timeframe,history_years,history_last,history_count,history_expected,coverage,model_trained_at,discovery_trained_at,optimizer_at,deep_research_at,status,message,last_error,version,pipeline_generation)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (key,tf,*[old.get(f,0 if f not in ('status','message','last_error','version') else '') for f in fields]))


def _unified_model_fresh(symbol,tf,years):
    """Determine freshness from model time + latest stored candle, not full 2y coverage."""
    key=canonical(symbol); tf=int(tf); now=int(time.time())
    state=_unified_state(key,tf)
    if not state: return False,"state-missing"
    if int(state.get('pipeline_generation') or 0) != UNIFIED_PIPELINE_GENERATION:
        return False,"legacy-pipeline"
    trained=max(int(state.get('model_trained_at') or 0),int(state.get('discovery_trained_at') or 0),int(state.get('optimizer_at') or 0))
    if trained<=0: return False,"never-trained"
    last_data=int(state.get('history_last') or 0)
    if last_data<=0: return False,"no-data"
    step=max(60,tf*60)
    # Data itself is stale only if the newest stored completed candle is far behind now.
    if now-last_data > max(step*6, UNIFIED_STALE_HOURS.get(tf,24)*3600):
        return False,"data-stale"
    # A model trained before newly downloaded candles should be retrained.
    if trained + step < last_data:
        return False,"new-data"
    stale_h=UNIFIED_STALE_HOURS.get(tf,24)
    if now-trained > stale_h*3600: return False,f"stale-{(now-trained)//3600}h"
    # Discovery summary is an optional enrichment layer. It must NOT make a
    # successfully trained unit appear stale. The old dashboard required this
    # file and therefore reported 0 ready even when the training_state was ready.
    try:
        summary=DISCOVERY_DIR/f"{key}_{tf}_summary.json"
        if not summary.exists():
            return True,"fresh-without-discovery"
    except Exception:
        return True,"fresh-without-discovery"
    return True,"fresh"

def _unified_prepare_history(symbol,tf,years):
    # Bootstrap a bounded recent window first. Full historical backfill is resumed
    # separately so the UI never waits for a multi-year 1m download.
    df=adaptive_fetch_and_store_history(symbol,tf,years=years,force=False,max_requests=UNIFIED_DATA_REQUESTS_PER_UNIT)
    audit=adaptive_history_coverage(symbol,tf,years)
    adaptive_save_history_audit(audit)
    last=int(audit.get('last') or 0)
    _unified_upsert_state(symbol,tf,history_years=float(years),history_last=last,
                          history_count=int(audit.get('count',0)),history_expected=int(audit.get('expected',0)),
                          coverage=float(audit.get('coverage_pct',0)),status='history_partial' if not audit.get('complete') else 'history',
                          message=('تاریخچه اولیه آماده؛ تکمیل پس‌زمینه فعال است' if not audit.get('complete') else 'تاریخچه کامل است'),version=str(UNIFIED_SCHEMA))
    return df,audit


def _unified_train_one(symbol,tf,years,deep=True,optimizer=True):
    key=canonical(symbol); tf=int(tf); now=int(time.time())
    _unified_upsert_state(key,tf,status='training',message='شروع آموزش',last_error='')
    df,audit=_unified_prepare_history(key,tf,years)
    if not audit.get('trainable'):
        _unified_upsert_state(key,tf,status='needs_data',message=f"داده کافی نیست: {audit.get('count',0)}/{ADAPTIVE_MIN_TRAIN_ROWS}",last_error='insufficient-history')
        return False,'insufficient-history'
    try:
        # The deep layer uses the currently available chronological dataframe.
        # Full requested coverage is not a prerequisite for first training.
        if deep and len(df)>=250:
            cp=_deep_load_checkpoint(key,tf,years=years)
            def save_method(name,payload,all_payload):
                cp['methods']=dict(all_payload); cp['last_completed']=name; cp['status']='running'; _deep_save_checkpoint(key,tf,cp,years=years)
            research=deep_research_dataframe(df,checkpoint=cp,checkpoint_save=save_method)
            report_dir=Path(DEEP_RESEARCH_DIR); report_dir.mkdir(parents=True,exist_ok=True)
            report=report_dir/f"{key}_{tf}m.json"
            payload={'schema':DEEP_RESEARCH_SCHEMA_VERSION,'symbol':key,'timeframe':tf,'years':years,'created_at':time.time(),'research':research}
            tmp=report.with_suffix('.json.tmp'); tmp.write_text(json.dumps(payload,ensure_ascii=False,default=str),encoding='utf-8'); os.replace(str(tmp),str(report))
            cp['methods']=dict(research); cp['status']='done'; cp['completed_at']=time.time(); _deep_save_checkpoint(key,tf,cp,years=years)
            _unified_upsert_state(key,tf,deep_research_at=now,status='research',message='تحلیل عمیق تکمیل شد')
        if optimizer and len(df)>=ADAPTIVE_MIN_TRAIN_ROWS:
            adaptive_optimize_symbol_tf(key,tf,years=years,force_history=False)
            _unified_upsert_state(key,tf,optimizer_at=int(time.time()),status='optimizer',message='بک‌تست OOS و بهینه‌سازی تکمیل شد')
        if len(df)>=DISCOVERY_MIN_ROWS:
            r=_discovery_train(key,tf,force=True)
            if r:
                _unified_upsert_state(key,tf,discovery_trained_at=int(time.time()),status='ml',message='مدل پیش‌بینی حرکت تکمیل شد')
            try:
                adaptive_train_ml_champions(key,tf,df)
                _unified_upsert_state(key,tf,model_trained_at=int(time.time()),pipeline_generation=UNIFIED_PIPELINE_GENERATION,status='ready',message='همه لایه‌ها آماده استفاده')
            except Exception as exc:
                _unified_upsert_state(key,tf,model_trained_at=int(time.time()),pipeline_generation=UNIFIED_PIPELINE_GENERATION,status='ready',message='Discovery آماده؛ Adaptive ML محدود',last_error=str(exc))
        else:
            _unified_upsert_state(key,tf,status='insufficient',message=f'کندل کافی نیست: {len(df)}')
            return False,'insufficient-data'
        return True,'updated'
    except Exception as exc:
        logging.exception('unified train failed %s/%s: %s',key,tf,exc)
        _unified_upsert_state(key,tf,status='error',message='خطا در آموزش',last_error=str(exc))
        return False,str(exc)


def _unified_save_prediction(symbol,tf,p):
    try:
        d=p.get('discovery') or {}
        now=int(time.time()); key=canonical(symbol)
        with _unified_db() as db:
            db.execute("""INSERT OR REPLACE INTO predictions
            (symbol,timeframe,ts,price,direction,prob_long,prob_short,prob_neutral,expected_move,confidence,quality,decision,model,trained_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (key,int(tf),now,float(p.get('price') or 0),str(d.get('direction') or p.get('discovery_direction') or 'NEUTRAL'),
             float(d.get('prob_long',p.get('prob_long',50))) * (100 if float(d.get('prob_long',p.get('prob_long',50)))<=1 else 1),
             float(d.get('prob_short',p.get('prob_short',50))) * (100 if float(d.get('prob_short',p.get('prob_short',50)))<=1 else 1),
             float(d.get('prob_neutral',p.get('prob_neutral',50))) * (100 if float(d.get('prob_neutral',p.get('prob_neutral',50)))<=1 else 1),
             float(p.get('predicted_signed_move_pct',d.get('primary_expected_return') or 0) or 0),
             float(p.get('confidence',0) or 0),float(p.get('signal_quality',d.get('quality',0)) or 0),
             str(p.get('decision','WAIT')),str(d.get('model') or 'Discovery'),int(_unified_state(key,tf).get('discovery_trained_at',0) if _unified_state(key,tf) else 0)))
    except Exception as exc:
        logging.debug('save prediction failed %s/%s: %s',symbol,tf,exc)


def _unified_predict_all(symbols=None,timeframes=None):
    """Generate predictions only for units that actually have a trained model.
    Never scan all symbols blindly during startup; this used to cause a second
    heavy API/ML workload immediately after the history pass.
    """
    symbols=list(symbols or _unified_symbols()); tfs=list(timeframes or UNIFIED_TIMEFRAMES); rows=[]
    ready=[]
    for sym in symbols:
        for tf in tfs:
            st=_unified_state(sym,tf) or {}
            trained=max(int(st.get('model_trained_at') or 0),int(st.get('discovery_trained_at') or 0))
            if trained>0 and int(st.get('history_count') or 0)>=DISCOVERY_MIN_ROWS:
                ready.append((sym,tf))
    for sym,tf in ready:
        try:
            p=signal_discovery_decision(sym,tf)
            _unified_save_prediction(sym,tf,p); rows.append((sym,tf,p))
        except Exception as exc:
            logging.debug('prediction failed %s/%s: %s',sym,tf,exc)
    return rows


def _unified_check_and_update(app_obj,years=None,force=False):
    """Startup gate: audit every symbol/TF, update only stale or incomplete units.
    The queue is resumable; a crash or close never invalidates completed units.
    """
    years=float(years or getattr(app_obj,'unified_years_var',tk.StringVar(value=str(UNIFIED_DEFAULT_YEARS))).get() or UNIFIED_DEFAULT_YEARS)
    years=max(1,min(10,years))
    try:
        prepare_symbol_mapping()
    except Exception:
        pass
    symbols=_unified_symbols(); tfs=list(UNIFIED_TIMEFRAMES)
    total=len(symbols)*len(tfs); completed=updated=skipped=failed=0
    run_id=None; started=int(time.time())
    _unified_init_db()
    with _unified_db() as db:
        cur=db.execute("INSERT INTO runs(started_at,status,total,completed,updated,skipped,failed,message) VALUES(?,?,?,?,?,?,?,?)",(started,'running',total,0,0,0,0,'شروع بررسی تازگی'))
        run_id=cur.lastrowid
    if hasattr(app_obj,'_unified_log'):
        app_obj._unified_ui_log(f'بررسی شروع شد: {len(symbols)} نماد × {len(tfs)} تایم‌فریم = {total} واحد')
    # Timeframe-major queue: high timeframe to low timeframe, all symbols per phase.
    phase_no=0
    for tf in tfs:
        phase_no+=1
        for i,sym in enumerate(symbols,1):
            if getattr(app_obj,'_unified_stop_requested',False):
                break
            completed+=1
            fresh,reason=_unified_model_fresh(sym,tf,years)
            if fresh and not force:
                skipped+=1
                _unified_upsert_state(sym,tf,status='ready',message='آموزش به‌روز است')
            else:
                ok,msg=_unified_train_one(sym,tf,years,deep=False,optimizer=False)
                if ok: updated+=1
                else: failed+=1
            if hasattr(app_obj,'_unified_progress'):
                pct=int(completed/max(1,total)*100)
                phase_pct=int(i/max(1,len(symbols))*100)
                try:
                    app_obj.root.after(0,lambda pct=pct,phase_pct=phase_pct,done_count=completed,sym=sym,tf=tf,total_count=total: app_obj._unified_set_progress(pct,f'فاز {tf}m | {phase_pct}% این تایم‌فریم | کل {done_count}/{total_count} | {sym}'))
                    if hasattr(app_obj,'_unified_phase_vars') and tf in app_obj._unified_phase_vars:
                        app_obj.root.after(0,lambda tf=tf,phase_pct=phase_pct: app_obj._unified_phase_vars[tf]['bar'].configure(value=phase_pct))
                        app_obj.root.after(0,lambda tf=tf,phase_pct=phase_pct: app_obj._unified_phase_vars[tf]['label'].configure(text=f'{tf_label(tf)}: {phase_pct}%'))
                except Exception: pass
            if completed%10==0:
                with _unified_db() as db:
                    db.execute("UPDATE runs SET completed=?,updated=?,skipped=?,failed=?,message=? WHERE id=?",(completed,updated,skipped,failed,f'{completed}/{total}',run_id))
        if getattr(app_obj,'_unified_stop_requested',False): break
    # Historical backfill is deliberately decoupled from first training. It runs
    # sequentially and keeps filling missing <=500-candle pages after the models exist.
    if not getattr(app_obj,'_unified_stop_requested',False):
        try:
            adaptive_start_background_backfill(symbols,tfs,years=years)
            if hasattr(app_obj,'_unified_ui_log'):
                app_obj._unified_ui_log('تکمیل تاریخچه کامل در پس‌زمینه آغاز شد؛ آموزش اولیه منتظر ۲ سال کامل نمی‌ماند.')
        except Exception as exc:
            logging.debug('unified background backfill failed: %s',exc)
    # Heavy research is deliberately separated from the first-pass training.
    # The user gets usable models/predictions first; deep research and OOS
    # optimization then run as a resumable low-pressure queue.
    if not getattr(app_obj,'_unified_stop_requested',False):
        def _heavy_worker():
            # NEVER compete with history acquisition. Wait until the downloader
            # has finished its current pass, then process completed units slowly.
            while not getattr(app_obj,'_unified_stop_requested',False):
                bt=globals().get('_adaptive_backfill_thread')
                if bt is None or not bt.is_alive():
                    break
                time.sleep(10.0)
            if getattr(app_obj,'_unified_stop_requested',False): return
            for hs in symbols:
                for ht in tfs:
                    if getattr(app_obj,'_unified_stop_requested',False): return
                    try:
                        st=_unified_state(hs,ht) or {}
                        if int(st.get('history_count') or 0) < ADAPTIVE_MIN_TRAIN_ROWS or float(st.get('coverage') or 0) < 99.9:
                            continue
                        _unified_train_one(hs,ht,years,deep=True,optimizer=True)
                        time.sleep(1.0)
                    except Exception as exc:
                        logging.debug('heavy unified research %s/%s: %s',hs,ht,exc)
                        time.sleep(2.0)
            try:
                app_obj.root.after(0,lambda: app_obj._unified_ui_log('داده‌ها تکمیل شد؛ تحلیل عمیق و بهینه‌سازی اکنون با صف کم‌فشار شروع شد.'))
            except Exception: pass
        threading.Thread(target=_heavy_worker,daemon=True,name='UnifiedHeavyResearchLowPressure').start()
    # Prediction scan is deliberately separate from training and can reuse every artifact.
    if not getattr(app_obj,'_unified_stop_requested',False):
        _unified_predict_all(symbols,tfs)
    status='stopped' if getattr(app_obj,'_unified_stop_requested',False) else 'done'
    msg=f'{status}: به‌روزرسانی={updated} | بدون نیاز={skipped} | خطا/ناکافی={failed}'
    with _unified_db() as db:
        db.execute("UPDATE runs SET finished_at=?,status=?,completed=?,updated=?,skipped=?,failed=?,message=? WHERE id=?",(int(time.time()),status,completed,updated,skipped,failed,msg,run_id))
    try:
        UNIFIED_RUN_FILE.write_text(json.dumps({'started_at':started,'finished_at':int(time.time()),'status':status,'total':total,'completed':completed,'updated':updated,'skipped':skipped,'failed':failed,'years':years,'symbols':len(symbols),'timeframes':list(tfs)},ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception: pass
    if hasattr(app_obj,'root'):
        try: app_obj.root.after(0,lambda: app_obj._unified_finished(msg))
        except Exception: pass



def tf_label(tf):
    return {1:"1m",5:"5m",15:"15m",30:"30m",60:"1h",240:"4h",1440:"1d"}.get(int(tf),f"{tf}m")


def _unified_check_all_history(self, years=None):
    """Read-only completeness audit for every symbol, in high-to-low TF phases.
    No network calls, no ML and no training. Results are persisted to the UI
    status panel so the user can verify exactly which timeframe is complete.
    """
    if getattr(self,'_unified_audit_thread',None) and self._unified_audit_thread.is_alive():
        messagebox.showinfo('بررسی تاریخچه','بررسی کامل بودن تاریخچه همین الان در حال اجراست.')
        return
    try: years=float(years or self.unified_years_var.get() or UNIFIED_DEFAULT_YEARS)
    except Exception: years=float(UNIFIED_DEFAULT_YEARS)
    symbols=list(self._unified_symbols())
    tfs=list(UNIFIED_TIMEFRAMES)
    self._unified_audit_stop=False
    def worker():
        total=max(1,len(symbols)*len(tfs)); done=0
        overall_complete=overall_trainable=overall_missing=0
        phase_rows=[]
        try:
            _unified_init_db(); self._unified_ui_log(f'شروع بررسی فقط-خواندنی: {len(symbols)} نماد × {len(tfs)} تایم‌فریم | ترتیب: 1d → 4h → 1h → 30m → 15m → 5m → 1m')
            self._unified_stage_set('history','active','ممیزی تاریخچه')
            for tf in tfs:
                if getattr(self,'_unified_audit_stop',False): break
                complete=trainable=missing=0
                for idx,sym in enumerate(symbols,1):
                    if getattr(self,'_unified_audit_stop',False): break
                    try:
                        audit=adaptive_history_coverage(sym,tf,years)
                        adaptive_save_history_audit(audit)
                        if audit.get('complete'): complete+=1
                        elif audit.get('trainable'): trainable+=1
                        else: missing+=1
                        _unified_upsert_state(sym,tf,history_years=float(years),history_last=int(audit.get('last') or 0),history_count=int(audit.get('count',0)),history_expected=int(audit.get('expected',0)),coverage=float(audit.get('coverage_pct',0)),status=('ready' if audit.get('complete') else 'ready_partial' if audit.get('trainable') else 'needs_data'),message=('تاریخچه کامل' if audit.get('complete') else 'داده کافی، تاریخچه ناقص' if audit.get('trainable') else 'داده ناکافی'))
                    except Exception as exc:
                        missing+=1
                        logging.debug('history audit %s/%s: %s',sym,tf,exc)
                    done+=1
                    phase_pct=int(idx/max(1,len(symbols))*100)
                    overall_pct=int(done/max(1,total)*100)
                    try:
                        self.root.after(0,lambda tf=tf,p=phase_pct,o=overall_pct,idx=idx: self._unified_set_progress(o,f'ممیزی {tf_label(tf)} | {p}% این فاز | کل {o}%'))
                        if hasattr(self,'_unified_phase_vars') and tf in self._unified_phase_vars:
                            self.root.after(0,lambda tf=tf,p=phase_pct: self._unified_phase_vars[tf]['bar'].configure(value=p))
                            self.root.after(0,lambda tf=tf,p=phase_pct: self._unified_phase_vars[tf]['label'].configure(text=f'{tf_label(tf)}: {p}%'))
                    except Exception: pass
                phase_rows.append((tf,complete,trainable,missing))
                overall_complete+=complete; overall_trainable+=trainable; overall_missing+=missing
                try:
                    self.root.after(0,lambda tf=tf,c=complete,t=trainable,m=missing: self._unified_phase_vars[tf]['label'].configure(text=f'{tf_label(tf)} ✓{c} / ◐{t} / !{m}'))
                except Exception: pass
                self._unified_ui_log(f'{tf_label(tf)}: کامل={complete} | قابل آموزش={trainable} | ناکافی={missing}')
            if not getattr(self,'_unified_audit_stop',False):
                self._unified_stage_set('history','done','ممیزی کامل شد')
            self._unified_ui_log(f'پایان ممیزی: کامل={overall_complete} | قابل آموزش={overall_trainable} | ناکافی={overall_missing} | کل واحدها={len(symbols)*len(tfs)}')
            try: self.root.after(0,self._unified_refresh_dashboard)
            except Exception: pass
        except Exception as exc:
            logging.exception('full history audit failed')
            self._unified_ui_log(f'خطای ممیزی: {exc}')
            try: self.root.after(0,lambda: self._unified_stage_set('history','error','خطای ممیزی'))
            except Exception: pass
    self._unified_audit_thread=threading.Thread(target=worker,daemon=True,name='UnifiedHistoryCompletenessAudit')
    self._unified_audit_thread.start()


def _unified_stage_set(self,key,state,text):
    if not hasattr(self,'unified_stage_labels') or key not in self.unified_stage_labels: return
    icon='✓' if state=='done' else '●' if state=='active' else '!' if state=='error' else '○'
    color='#22c55e' if state=='done' else '#fbbf24' if state=='active' else '#ef4444' if state=='error' else '#64748b'
    def apply():
        try: self.unified_stage_labels[key].config(text=icon,fg=color); self.unified_stage_texts[key].config(text=text,fg=color)
        except Exception: pass
    try: self.root.after(0,apply)
    except Exception: apply()


def _unified_build_tab(self):
    tab=_make_scrollable_tab(self.notebook,bg='#07111f'); self.unified_center_tab=tab
    self.notebook.add(tab._scroll_outer,text='🧠 مرکز هوش بازار')
    head=tk.Frame(tab,bg='#0f1b2d'); head.pack(fill='x',padx=8,pady=8)
    tk.Label(head,text='مرکز هوش بازار — PIPELINE V12 | تحلیل عمیق + اجرای هوشمند + ساختار بازار + ریسک/سرمایه + بک‌تست + یادگیری/بهینه‌سازی + ML + پیش‌بینی',bg='#0f1b2d',fg='#67e8f9',font=('Tahoma',13,'bold')).pack(side='left',padx=8)
    self.unified_status=tk.Label(head,text='آماده — منتظر فرمان کاربر',bg='#0f1b2d',fg='#fbbf24',font=('Tahoma',10,'bold')); self.unified_status.pack(side='left',padx=15)

    ctl=tk.Frame(tab,bg='#0b1626'); ctl.pack(fill='x',padx=8,pady=(0,6))
    tk.Label(ctl,text='تاریخچه هدف (سال):',bg='#0b1626',fg='white').pack(side='left',padx=(8,3))
    self.unified_years_var=tk.StringVar(value=str(UNIFIED_DEFAULT_YEARS))
    ttk.Combobox(ctl,textvariable=self.unified_years_var,values=['1','2','3','5','10'],state='readonly',width=5).pack(side='left')
    tk.Button(ctl,text='📁 مسیر ذخیره',command=self._unified_choose_storage,bg='#334155',fg='white').pack(side='left',padx=6)
    self.unified_path_label=tk.Label(ctl,text=str(UNIFIED_STORAGE_DIR),bg='#0b1626',fg='#93c5fd',anchor='w'); self.unified_path_label.pack(side='left',fill='x',expand=True,padx=8)
    tk.Button(ctl,text='▶ بررسی و بروزرسانی',command=self._unified_start_update,bg='#059669',fg='white',font=('Tahoma',9,'bold')).pack(side='right',padx=3)
    tk.Button(ctl,text='☑ بررسی کامل همه تایم‌فریم‌ها',command=self._unified_check_all_history,bg='#0ea5e9',fg='white',font=('Tahoma',9,'bold')).pack(side='right',padx=3)
    tk.Button(ctl,text='⟳ آموزش اجباری',command=lambda:self._unified_start_update(True),bg='#b45309',fg='white').pack(side='right',padx=3)
    tk.Button(ctl,text='■ توقف',command=self._unified_stop,bg='#dc2626',fg='white').pack(side='right',padx=3)

    stage=tk.Frame(tab,bg='#07111f'); stage.pack(fill='x',padx=8,pady=4)
    self.unified_stage_labels={}; self.unified_stage_texts={}
    for k,t in [('history','۱ تاریخچه'),('deep','۲ تحلیل عمیق'),('structure','۳ ساختار بازار'),('risk','۴ ریسک/سرمایه'),('backtest','۵ بک‌تست/OOS'),('opt','۶ بهینه‌سازی'),('ml','۷ ML/کالیبراسیون'),('predict','۸ پیش‌بینی')]:
        b=tk.Frame(stage,bg='#111c2d',bd=1,relief='solid'); b.pack(side='left',fill='x',expand=True,padx=2)
        tk.Label(b,text=t,bg='#111c2d',fg='#cbd5e1',font=('Tahoma',8,'bold')).pack(pady=(3,0))
        v=tk.Label(b,text='○',bg='#111c2d',fg='#64748b',font=('Tahoma',10,'bold')); v.pack(pady=(1,0)); self.unified_stage_labels[k]=v
        tx=tk.Label(b,text='آماده',bg='#111c2d',fg='#64748b',font=('Tahoma',7)); tx.pack(pady=(0,3)); self.unified_stage_texts[k]=tx

    phase=tk.LabelFrame(tab,text='صف دریافت و ممیزی تاریخچه — از تایم‌فریم بالاتر به پایین‌تر',bg='#07111f',fg='#93c5fd',font=('Tahoma',9,'bold')); phase.pack(fill='x',padx=8,pady=4)
    self._unified_phase_vars={}
    for tf in UNIFIED_TIMEFRAMES:
        cell=tk.Frame(phase,bg='#0b1626'); cell.pack(side='left',fill='x',expand=True,padx=2,pady=3)
        lab=tk.Label(cell,text=f'{tf_label(tf)}: 0%',bg='#0b1626',fg='#cbd5e1',font=('Tahoma',8,'bold')); lab.pack(fill='x')
        bar=ttk.Progressbar(cell,mode='determinate',maximum=100); bar.pack(fill='x',padx=2,pady=(2,1))
        self._unified_phase_vars[tf]={'label':lab,'bar':bar}

    stats=tk.Frame(tab,bg='#07111f'); stats.pack(fill='x',padx=8,pady=5)
    self.unified_cards={}
    for k,t in [('symbols','نماد'),('units','واحد'),('ready','مدل آماده'),('training','قابل آموزش'),('needs_data','نیازمند داده'),('coverage','پوشش میانگین'),('pred','پیش‌بینی')]:
        f=tk.Frame(stats,bg='#111c2d',bd=1,relief='solid'); f.pack(side='left',fill='x',expand=True,padx=2)
        tk.Label(f,text=t,bg='#111c2d',fg='#94a3b8').pack(); v=tk.Label(f,text='—',bg='#111c2d',fg='white',font=('Tahoma',11,'bold')); v.pack(pady=3); self.unified_cards[k]=v

    body=tk.PanedWindow(tab,orient='vertical',bg='#07111f',sashwidth=6); body.pack(fill='both',expand=True,padx=8,pady=5)
    top=tk.Frame(body,bg='#0b1626'); body.add(top,stretch='always')
    cols=('symbol','tf','coverage','status','direction','long','short','move','confidence','quality','decision','trained')
    titles={'symbol':'نماد','tf':'TF','coverage':'پوشش','status':'وضعیت','direction':'جهت','long':'Long%','short':'Short%','move':'حرکت پیش‌بینی','confidence':'اعتماد','quality':'کیفیت OOS','decision':'تصمیم','trained':'آخرین آموزش'}
    self.unified_table=ttk.Treeview(top,columns=cols,show='headings')
    for c in cols:self.unified_table.heading(c,text=titles[c]);self.unified_table.column(c,width=105,anchor='center')
    tv=ttk.Scrollbar(top,orient='vertical',command=self.unified_table.yview); th=ttk.Scrollbar(top,orient='horizontal',command=self.unified_table.xview)
    self.unified_table.configure(yscrollcommand=tv.set,xscrollcommand=th.set); self.unified_table.grid(row=0,column=0,sticky='nsew');tv.grid(row=0,column=1,sticky='ns');th.grid(row=1,column=0,sticky='ew');top.grid_rowconfigure(0,weight=1);top.grid_columnconfigure(0,weight=1);_install_scroll_support(self.unified_table)
    self.unified_table.tag_configure('long',background='#14532d',foreground='white');self.unified_table.tag_configure('short',background='#7f1d1d',foreground='white');self.unified_table.tag_configure('wait',background='#1e293b',foreground='#cbd5e1');self.unified_table.tag_configure('stale',background='#78350f',foreground='white')
    bottom=tk.Frame(body,bg='#07111f'); body.add(bottom,stretch='always')
    self.unified_summary=tk.Text(bottom,height=9,bg='#020617',fg='#dbeafe',font=('Consolas',9),wrap='word'); sv=ttk.Scrollbar(bottom,orient='vertical',command=self.unified_summary.yview); self.unified_summary.configure(yscrollcommand=sv.set); self.unified_summary.pack(side='left',fill='both',expand=True);sv.pack(side='right',fill='y')
    self.unified_progress=ttk.Progressbar(tab,mode='determinate',maximum=100);self.unified_progress.pack(fill='x',padx=8,pady=(0,3))
    self.unified_log=tk.Text(tab,height=6,bg='#020617',fg='#94a3b8',font=('Consolas',8),wrap='word');self.unified_log.pack(fill='x',padx=8,pady=(0,6))
    self._unified_stop_requested=False
    # Never perform the first full dashboard scan synchronously while the Tk
    # window is being constructed. With hundreds of symbols this used to make
    # Windows mark the process as "Not Responding" before the mainloop even
    # had a chance to paint the window. The first refresh is scheduled after
    # the event loop starts.
    self.root.after(150, self._unified_refresh_dashboard)
    self.root.after(UNIFIED_DASHBOARD_REFRESH_MS, self._unified_dashboard_tick)
    # IMPORTANT: heavy history/ML processing is MANUAL. Do not start it on application launch.
    # The user must explicitly press '▶ بررسی و بروزرسانی' or '☑ بررسی کامل همه تایم‌فریم‌ها'.


def _unified_set_progress(self,pct,msg):
    try:self.unified_progress['value']=pct;self.unified_status.config(text=msg)
    except Exception:pass


def _unified_ui_log(self,msg):
    try:
        def put():
            self.unified_log.insert('end',f'[{time.strftime("%H:%M:%S")}] {msg}\n');self.unified_log.see('end')
        self.root.after(0,put)
    except Exception:pass


def _unified_choose_storage(self):
    try:
        folder=filedialog.askdirectory(parent=self.root,title='مسیر دائمی ذخیره تاریخچه و آموزش را انتخاب کنید',initialdir=str(UNIFIED_STORAGE_DIR))
        if folder and _unified_set_storage(folder):
            self.unified_path_label.config(text=str(UNIFIED_STORAGE_DIR));self._unified_ui_log(f'مسیر ذخیره تغییر کرد: {UNIFIED_STORAGE_DIR}');self._unified_refresh_dashboard()
    except Exception as exc: messagebox.showerror('مسیر ذخیره',str(exc))


def _unified_stop(self):
    self._unified_stop_requested=True;self.unified_status.config(text='درخواست توقف ثبت شد؛ واحد جاری تکمیل می‌شود...')


def _unified_start_update(self,force=False):
    if getattr(self,'_unified_thread',None) and self._unified_thread.is_alive():
        messagebox.showinfo('مرکز هوش بازار','یک فرآیند در حال اجراست.')
        return
    try: years=float(self.unified_years_var.get())
    except Exception: years=UNIFIED_DEFAULT_YEARS
    self._unified_stop_requested=False
    self._unified_ui_log(f'شروع دستی Pipeline با فرمان کاربر | تاریخچه هدف: {years:g} سال | ترتیب: 1d → 4h → 1h → 30m → 15m → 5m → 1m')
    self._unified_thread=threading.Thread(target=_unified_check_and_update,args=(self,years,force),daemon=True,name='UnifiedMarketIntelligence');self._unified_thread.start()


def _unified_dashboard_tick(self):
    """Continuously reconcile the visible dashboard with SQLite state.

    The previous build refreshed only at startup and when the whole 40-unit
    queue finished, so a live run looked frozen for hours. This timer is GUI-
    thread-only and reads SQLite; it never performs network/ML work.
    """
    try:
        self._unified_refresh_dashboard()
    except Exception as exc:
        logging.debug("unified dashboard tick: %s", exc)
    try:
        self.root.after(UNIFIED_DASHBOARD_REFRESH_MS, self._unified_dashboard_tick)
    except Exception:
        pass


def _unified_startup_check(self):
    """Startup hook intentionally does NOT launch network/history/ML work.

    The unified pipeline is user-triggered only. This prevents the application
    from consuming CPU/network/disk immediately after launch. The dashboard
    may refresh its already-persisted SQLite state, but no worker is started.
    """
    self._unified_ui_log('برنامه آماده است؛ پردازش سنگین خودکار شروع نمی‌شود. برای شروع، دکمه «▶ بررسی و بروزرسانی» را بزنید.')
    try:
        self.unified_status.config(text='آماده — منتظر فرمان کاربر')
    except Exception:
        pass


def _unified_finished(self,msg):
    self.unified_status.config(text=msg);self._unified_refresh_dashboard();self._unified_ui_log(msg)
    for k in ('history','deep','structure','risk','backtest','opt','ml','predict'):
        try:self.unified_stage_labels[k].config(text='✓',fg='#22c55e')
        except Exception:pass


def _unified_refresh_dashboard(self):
    """Reconcile the dashboard from one SQLite snapshot.

    V11 fixes the misleading counters: ``ready`` is only a fresh trained unit;
    ``training`` means enough candles exist but no fresh model exists;
    ``needs_data`` means the minimum history is not present.  Coverage is the
    mean candle coverage across the CURRENT 7 timeframes, not ``complete units /
    total``.  This prevents an old group of 40 models from making the dashboard
    look as if the whole market is ready.
    """
    try:
        syms=_unified_symbols(); tfs=list(UNIFIED_TIMEFRAMES)
        total=len(syms)*len(tfs); now_s=int(time.time()); now_ms=int(time.time()*1000)
        with _unified_db() as db:
            states=db.execute(
                'SELECT symbol,timeframe,coverage,status,model_trained_at,discovery_trained_at,'
                'history_last,history_count,history_expected,history_years,version,pipeline_generation FROM training_state'
            ).fetchall()
            preds=db.execute(
                'SELECT symbol,timeframe,direction,prob_long,prob_short,expected_move,confidence,quality,decision,model'
                ' FROM predictions'
            ).fetchall()
        sm={(str(r[0]),int(r[1])):r for r in states}
        pm={(str(r[0]),int(r[1])):r for r in preds}
        ready=training=needs_data=updating=stale=covered=0
        coverage_sum=0.0; rows=[]
        for sym in syms:
            for tf in tfs:
                st=sm.get((sym,tf)); pr=pm.get((sym,tf))
                if st:
                    cov=max(0.0,min(100.0,float(st[2] or 0))); status=str(st[3] or 'pending')
                    trained=max(int(st[4] or 0),int(st[5] or 0)); last_data=int(st[6] or 0)
                    data_rows=int(st[7] or 0); expected=int(st[8] or 0); generation=int(st[11] or 0)
                else:
                    cov=0.0; status='needs_data'; trained=0; last_data=0; data_rows=0; expected=0; generation=0
                display_status=status
                coverage_sum += cov
                if cov>=99.99: covered += 1
                step=max(60,int(tf)*60); stale_h=UNIFIED_STALE_HOURS.get(tf,24)
                fresh=False
                if trained>0 and generation==UNIFIED_PIPELINE_GENERATION and last_data>0:
                    data_stale=(now_s-last_data)>max(step*6,stale_h*3600)
                    model_stale=(now_s-trained)>stale_h*3600
                    new_data=(trained+step<last_data)
                    fresh=not data_stale and not model_stale and not new_data
                if generation != UNIFIED_PIPELINE_GENERATION and trained>0:
                    stale+=1
                    display_status='legacy_model'
                elif status in ('updating','training','history','history_partial','research','optimizer','ml'):
                    updating+=1
                    display_status=status
                elif fresh and generation==UNIFIED_PIPELINE_GENERATION and trained>0 and status in ('ready','ready_partial','ml','optimizer','research'):
                    ready+=1
                elif data_rows>=ADAPTIVE_MIN_TRAIN_ROWS and trained<=0:
                    training+=1
                elif trained>0 and not fresh:
                    stale+=1
                else:
                    needs_data+=1
                if pr:
                    direction=str(pr[2] or 'WAIT'); long=float(pr[3] or 0); short=float(pr[4] or 0)
                    move=float(pr[5] or 0); conf=float(pr[6] or 0); qual=float(pr[7] or 0); decision=str(pr[8] or 'WAIT')
                    vals=(sym,f'{tf}m',f'{cov:.1f}%',display_status if 'display_status' in locals() else status,direction,f'{long:.1f}',f'{short:.1f}',f'{move:+.3f}%',f'{conf:.1f}',f'{qual:.1f}',decision,time.strftime('%Y-%m-%d %H:%M',time.localtime(trained)) if trained else '—')
                else:
                    vals=(sym,f'{tf}m',f'{cov:.1f}%',status,'—','—','—','—','—','—','—','—')
                rows.append(vals)
        avg_cov=coverage_sum/max(1,total)
        pred_count=len(pm)
        signature=(ready,training,needs_data,updating,stale,covered,pred_count,round(avg_cov,2),tuple((r[0],r[1],r[2],r[3],r[10]) for r in rows[:120]))
        last_sig=getattr(self,'_unified_dashboard_signature',None)
        last_rebuild=int(getattr(self,'_unified_dashboard_rebuild_ms',0))
        must_rebuild=(signature!=last_sig and (now_ms-last_rebuild)>=UNIFIED_DASHBOARD_REBUILD_MS)
        self.unified_cards['symbols'].config(text=str(len(syms)))
        self.unified_cards['units'].config(text=str(total))
        self.unified_cards['ready'].config(text=str(ready))
        self.unified_cards['training'].config(text=str(training))
        self.unified_cards['needs_data'].config(text=str(needs_data+stale+updating))
        self.unified_cards['coverage'].config(text=f'{avg_cov:.2f}%')
        self.unified_cards['pred'].config(text=str(pred_count))
        self.unified_status.config(text=f'آماده: {ready} | قابل آموزش: {training} | در حال پردازش: {updating} | نیازمند داده/بروزرسانی: {needs_data+stale}')
        if must_rebuild and hasattr(self,'unified_table'):
            display=sorted(rows,key=lambda x:(0 if x[3] in ('training','updating','needs_data','pending') else 1,x[0],x[1]))[:UNIFIED_TABLE_MAX_ROWS]
            self.unified_table.delete(*self.unified_table.get_children())
            for vals in display:
                direction=str(vals[4]); decision=str(vals[10])
                tag='long' if 'LONG' in direction or decision.startswith('BUY') else 'short' if 'SHORT' in direction or decision.startswith('SELL') else 'wait'
                self.unified_table.insert('', 'end', values=vals, tags=(tag,))
            self._unified_dashboard_signature=signature; self._unified_dashboard_rebuild_ms=now_ms
        summary_sig=(total,ready,training,needs_data,updating,stale,pred_count,round(avg_cov,2),covered)
        if summary_sig != getattr(self,'_unified_summary_signature',None):
            self.unified_summary.delete('1.0','end')
            self.unified_summary.insert('end',
                f'ذخیره دائمی: {UNIFIED_STORAGE_DIR}\n'
                f'واحدها: {total} | مدل آماده: {ready} | قابل آموزش: {training} | در حال پردازش: {updating} | '
                f'نیازمند داده/بروزرسانی: {needs_data+stale} | پیش‌بینی: {pred_count}\n'
                f'پوشش میانگین تاریخچه: {avg_cov:.2f}% | واحدهای دارای پوشش کامل: {covered}/{total}\n'
                f'نسل مدل موردنیاز: {UNIFIED_PIPELINE_GENERATION} | مدل‌های نسل قدیمی در «آماده» شمرده نمی‌شوند.\n\n'
                'توجه: «مدل آماده» فقط مدل تازه و آموزش‌دیده را شمارش می‌کند. داده کافی بدون مدل در «قابل آموزش» است. '
                'باز کردن برنامه Pipeline یادگیری را شروع نمی‌کند؛ شروع فقط با فرمان کاربر انجام می‌شود.\n'
                'سیگنال‌های BUY/SELL در لاگ عمومی موتور اسکالپر مستقل از Pipeline یادگیری هستند.')
            self._unified_summary_signature=summary_sig
    except Exception as exc:
        logging.debug('unified refresh dashboard V12: %s',exc)
        try:
            self.unified_status.config(text=f'خطای داشبورد: {exc}')
        except Exception:
            pass


# Preserve every existing application tab. The Unified Market Intelligence
# center is an additional orchestration tab; it does not replace or remove
# Overview, Smart Execution, Market Structure, Risk, Backtest, Learning, API,
# Trade Monitor, Evidence, or any other existing tab.
try:
    _legacy_build_ui_for_unified = NobitexScalperPro._build_ui
    def _build_ui_with_unified_center(self):
        _legacy_build_ui_for_unified(self)
        try:
            # Add one dedicated orchestration tab after the mature UI is built.
            _unified_build_tab(self)
            self.notebook.select(getattr(self.unified_center_tab, "_scroll_outer", self.unified_center_tab))
        except Exception as exc:
            logging.exception('unified center UI build failed: %s', exc)
    NobitexScalperPro._build_ui=_build_ui_with_unified_center
    NobitexScalperPro._unified_choose_storage=_unified_choose_storage
    NobitexScalperPro._unified_stop=_unified_stop
    NobitexScalperPro._unified_start_update=_unified_start_update
    NobitexScalperPro._unified_check_all_history=_unified_check_all_history
    NobitexScalperPro._unified_stage_set=_unified_stage_set
    NobitexScalperPro._unified_startup_check=_unified_startup_check
    NobitexScalperPro._unified_dashboard_tick=_unified_dashboard_tick
    NobitexScalperPro._unified_set_progress=_unified_set_progress
    NobitexScalperPro._unified_ui_log=_unified_ui_log
    NobitexScalperPro._unified_finished=_unified_finished
    NobitexScalperPro._unified_refresh_dashboard=_unified_refresh_dashboard
except Exception as exc:
    logging.debug('unified UI binding failed: %s',exc)



# ============================================================
# V13 OPERATIONAL PATCH
# ============================================================
# Goals:
# - live Overview price/change refresh independent from the analysis rotation
# - normal green BUY/SELL signals are eligible for the same simulated auto-trade
#   path as strong/gold signals, subject to explicit risk gates
# - clean current-only display settings (legacy layout file is ignored)
# - richer live trade chart
# - Unified Market Intelligence starts automatically in a low-pressure,
#   resumable, timeframe-major queue: 1m -> 5m -> 15m -> 30m -> 1h -> 4h -> 1d
# - dashboard cards use one coherent state machine

V13_OPERATIONAL_VERSION = "V13_MARKET_CENTER_OPERATIONAL"
V13_TIMEFRAMES = (1, 5, 15, 30, 60, 240, 1440)
# Keep both research engines on the same phase order.
try:
    ADAPTIVE_RESEARCH_TIMEFRAMES = list(V13_TIMEFRAMES)
except Exception:
    pass
try:
    UNIFIED_TIMEFRAMES = V13_TIMEFRAMES
    UNIFIED_PIPELINE_GENERATION = max(int(globals().get("UNIFIED_PIPELINE_GENERATION", 11)) + 1, 13)
except Exception:
    pass
UNIFIED_PIPELINE_VERSION = V13_OPERATIONAL_VERSION
V13_AUTO_START_UNIFIED = True
V13_QUOTE_REFRESH_SEC = 8.0
V13_CENTER_AUTO_DELAY_MS = 2500
V13_NORMAL_AUTO_TRADE_CONF = 68.0
V13_NORMAL_AUTO_TRADE_RR = 1.50
V13_STRONG_AUTO_TRADE_CONF = 72.0
V13_STRONG_AUTO_TRADE_RR = 1.80


def _v13_fetch_market_quotes():
    """Return a richer bulk quote snapshot without touching Tkinter."""
    url = urljoin(API_BASE, "/market/stats")
    try:
        r = get_session().get(url, timeout=max(4, min(REQUEST_TIMEOUT, 8)))
        if r.status_code != 200:
            return {}
        j = r.json()
        stats = j.get("stats") or j.get("result") or j.get("data") or j.get("markets") or []
        out = {}
        if isinstance(stats, dict):
            iterator = stats.items()
        else:
            iterator = []
            for item in stats if isinstance(stats, list) else []:
                if isinstance(item, dict):
                    k = item.get("symbol") or item.get("pair") or item.get("id")
                    if k:
                        iterator.append((k, item))
        for raw_key, item in iterator:
            if not isinstance(item, dict):
                continue
            key = api_key_to_canonical(raw_key)
            price = item.get("latest") or item.get("lastTradePrice") or item.get("price") or item.get("mark")
            try:
                price = float(price)
            except Exception:
                continue
            if price <= 0:
                continue
            # Different API revisions expose different daily-change names.
            ch = None
            for ck in ("dayChangePercent", "changePercent", "dailyChangePercent", "change", "percentChange"):
                if item.get(ck) is not None:
                    try:
                        ch = float(item.get(ck))
                        # Some endpoints return 0.0123 for 1.23%.
                        if abs(ch) < 1.0:
                            ch *= 100.0
                    except Exception:
                        ch = None
                    if ch is not None:
                        break
            out[key] = {"price": price, "change": ch}
        return out
    except Exception as exc:
        logging.debug("V13 quote snapshot failed: %s", exc)
        return {}


def _v13_apply_quotes(self, snapshot):
    """Update price/change cells only; never re-run analysis on the UI thread."""
    try:
        if not snapshot:
            return
        changed = {}
        with state_lock:
            for sym, q in snapshot.items():
                key = canonical(sym)
                p = safe_float(q.get("price"))
                if p <= 0:
                    continue
                old = safe_float(valid_symbols_map.get(key))
                valid_symbols_map[key] = p
                if q.get("change") is not None:
                    ch = float(q.get("change"))
                elif old > 0:
                    ch = (p - old) / old * 100.0
                else:
                    ch = 0.0
                last_delta_cache[key] = ch
                changed[key] = (p, ch)
        # Touch only visible rows and preserve all other columns.
        if not hasattr(self, "main_table"):
            return
        for iid in self.main_table.get_children():
            vals = list(self.main_table.item(iid).get("values", []))
            if not vals:
                continue
            key = canonical(vals[0])
            item = changed.get(key)
            if not item:
                continue
            p, ch = item
            if len(vals) >= 3:
                vals[1] = format_price(p)
                vals[2] = f"{ch:+.2f}%"
                self.main_table.item(iid, values=vals)
    except Exception as exc:
        logging.debug("V13 apply quotes failed: %s", exc)


def _v13_quote_loop(self):
    """Dedicated bulk quote loop. It is deliberately independent of candle analysis."""
    while getattr(self, "running", False) and not getattr(self, "_v13_quote_stop", False):
        try:
            snap = _v13_fetch_market_quotes()
            if snap:
                try:
                    self.root.after(0, lambda s=snap: _v13_apply_quotes(self, s))
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(V13_QUOTE_REFRESH_SEC)


def _v13_start_quote_loop(self):
    if getattr(self, "_v13_quote_thread", None) and self._v13_quote_thread.is_alive():
        return
    self._v13_quote_stop = False
    self._v13_quote_thread = threading.Thread(target=_v13_quote_loop, args=(self,), daemon=True, name="V13BulkQuotes")
    self._v13_quote_thread.start()


def _v13_clean_layout_build(self):
    """Build display settings from the current UI only; never expose removed/merged controls."""
    try:
        self._layout_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_layout_settings_v13.json")
        self._layout_tabs = {}
        self._layout_tables = {}
        self._layout_column_vars = {}
        self._layout_width_vars = {}
        # Current notebook tabs at build time.
        for idx, tab_id in enumerate(self.notebook.tabs()):
            try:
                widget = self.notebook.nametowidget(tab_id)
                label = self.notebook.tab(tab_id, "text")
                self._layout_tabs[str(tab_id)] = {"widget": widget, "text": label, "index": idx}
            except Exception:
                pass
        table_specs = [
            ("main_table", "نمای کلی"),
            ("ai_table", "مرکز معاملات هوشمند / AI"),
            ("smart_table", "اجرای هوشمند"),
            ("trade_monitor_table", "مانیتور معاملات"),
            ("risk_table", "ریسک و سرمایه"),
            ("deep_table", "آزمایشگاه تحلیل عمیق"),
            ("unified_table", "مرکز هوش بازار"),
            ("evidence_table", "Evidence / ارزیابی سیگنال"),
        ]
        for attr, title in table_specs:
            tree = getattr(self, attr, None)
            if tree is not None:
                self._layout_tables[attr] = {"widget": tree, "title": title}
        tab = _make_scrollable_tab(self.notebook, bg="#0f172a")
        self._layout_settings_tab = tab
        self.notebook.add(tab._scroll_outer, text="⚙️ تنظیمات نمایش")
        top = tk.Frame(tab, bg="#1e293b"); top.pack(fill="x", padx=8, pady=8)
        tk.Label(top, text="تنظیمات نمایش — فقط اجزای فعال نسخه جدید", bg="#1e293b", fg="white", font=("Tahoma",13,"bold")).pack(side="left", padx=10)
        tk.Button(top, text="ذخیره", command=self._save_layout_settings, bg="#10b981", fg="white", width=10).pack(side="right", padx=4)
        tk.Button(top, text="بازنشانی نسخه جدید", command=self._reset_layout_settings, bg="#ef4444", fg="white", width=15).pack(side="right", padx=4)
        body=tk.Frame(tab,bg="#0f172a"); body.pack(fill="both",expand=True,padx=8,pady=4)
        tabs_box=tk.LabelFrame(body,text="تب‌های فعال",bg="#111827",fg="#93c5fd",font=("Tahoma",10,"bold"),padx=8,pady=8); tabs_box.pack(fill="x",pady=(0,8))
        self._layout_tab_vars={}
        for key,info in self._layout_tabs.items():
            v=tk.BooleanVar(value=True); self._layout_tab_vars[key]=v
            tk.Checkbutton(tabs_box,text=info["text"],variable=v,command=self._apply_tab_visibility,bg="#111827",fg="white",selectcolor="#1e293b",activebackground="#111827",activeforeground="white").pack(side="left",padx=6,pady=3)
        canvas=tk.Canvas(body,bg="#0f172a",highlightthickness=0); sb=ttk.Scrollbar(body,orient="vertical",command=canvas.yview); inner=tk.Frame(canvas,bg="#0f172a")
        inner.bind("<Configure>",lambda e:canvas.configure(scrollregion=canvas.bbox("all"))); canvas.create_window((0,0),window=inner,anchor="nw"); canvas.configure(yscrollcommand=sb.set); canvas.pack(side="left",fill="both",expand=True); sb.pack(side="right",fill="y")
        for attr,info in self._layout_tables.items():
            tree=info["widget"]; box=tk.LabelFrame(inner,text=info["title"],bg="#111827",fg="#93c5fd",font=("Tahoma",10,"bold"),padx=8,pady=6); box.pack(fill="x",padx=4,pady=5)
            cf=tk.Frame(box,bg="#111827"); cf.pack(fill="x")
            for i,col in enumerate(tree["columns"]):
                title=tree.heading(col,"text") or col; var=tk.BooleanVar(value=True); self._layout_column_vars[(attr,col)]=var
                tk.Checkbutton(cf,text=title,variable=var,command=lambda a=attr:self._apply_table_layout(a),bg="#111827",fg="white",selectcolor="#1e293b",activebackground="#111827",activeforeground="white",anchor="w").grid(row=i//4,column=i%4,sticky="w",padx=5,pady=2)
                wv=tk.IntVar(value=int(tree.column(col,"width") or 110)); self._layout_width_vars[(attr,col)]=wv
                tk.Spinbox(cf,from_=50,to=500,increment=10,width=5,textvariable=wv,command=lambda a=attr:self._apply_table_layout(a),bg="#0b1220",fg="white",insertbackground="white").grid(row=i//4,column=i%4,sticky="e",padx=(0,8),pady=2)
            tk.Button(box,text="همه ستون‌ها",command=lambda a=attr:self._show_all_table_columns(a),bg="#2563eb",fg="white").pack(side="left",padx=4,pady=4)
            tk.Button(box,text="فقط ستون اول",command=lambda a=attr:self._hide_all_but_first(a),bg="#475569",fg="white").pack(side="left",padx=4,pady=4)
        self._layout_info_label=tk.Label(tab,text="این صفحه فقط تنظیمات مربوط به رابط کاربری فعال V13 را نشان می‌دهد. تنظیمات قدیمی نادیده گرفته می‌شوند.",bg="#0f172a",fg="#94a3b8",font=("Tahoma",9)); self._layout_info_label.pack(fill="x",padx=10,pady=5)
    except Exception as exc:
        logging.exception("V13 clean layout build failed: %s",exc)


def _v13_clean_apply_saved_layout(self):
    try:
        # Deliberately ignore the obsolete ui_layout_settings.json.
        for attr in self._layout_tables:
            self._apply_table_layout(attr)
        self._apply_tab_visibility()
    except Exception as exc:
        logging.debug("V13 apply layout failed: %s",exc)


def _v13_reset_layout(self):
    try:
        for v in self._layout_tab_vars.values(): v.set(True)
        for (attr,c),v in self._layout_column_vars.items():
            v.set(True)
            try:self._layout_width_vars[(attr,c)].set(110)
            except Exception:pass
        for attr in self._layout_tables:self._apply_table_layout(attr)
        self._apply_tab_visibility(); self._save_layout_settings()
    except Exception:pass


def _v13_pretty_trade_chart(self, selected=None):
    """Fancier live trade chart: grid, glow, price line, levels, badges and compact metrics."""
    try:
        c=self.trade_chart; c.delete("all")
        if selected is None:
            key=getattr(self,"_trade_monitor_selected_key",None)
            with state_lock: orders=list(auto_trade_log)
            selected=next((o for o in orders if key is not None and self._trade_key(o)==key),None)
            if selected is None: selected=next((o for o in orders if o.get("result") is None),None)
        w=max(920,c.winfo_width()); h=max(470,c.winfo_height())
        c.create_rectangle(0,0,w,h,fill="#050b14",outline="")
        if not selected:
            c.create_text(w/2,h/2,text="LIVE MARKET MONITOR  •  هیچ معامله‌ای برای نمایش وجود ندارد",fill="#64748b",font=("Tahoma",15,"bold")); return
        hist=[x for x in list(selected.get("price_history") or []) if safe_float(x.get("price"))>0]
        entry=safe_float(selected.get("entry_price")); cur=safe_float(selected.get("current_price")) or entry
        if not hist: hist=[{"ts":time.time()-60,"price":entry},{"ts":time.time(),"price":cur}]
        if len(hist)<2: hist.append({"ts":time.time(),"price":cur})
        levels=[("ENTRY",entry,"#f59e0b"),("TP1",safe_float(selected.get("take_profit_1")),"#22c55e"),("TP2",safe_float(selected.get("take_profit_2")),"#86efac"),("SL",safe_float(selected.get("stop_loss_price")),"#ef4444"),("TRAIL",safe_float(selected.get("trailing_stop")),"#38bdf8")]
        prices=[safe_float(x.get("price")) for x in hist if safe_float(x.get("price"))>0]+[v for _,v,_ in levels if v>0]
        lo=min(prices); hi=max(prices); pad=max((hi-lo)*.10,lo*.001); lo-=pad; hi+=pad
        left,top,right,bottom=58,52,w-255,h-54
        c.create_text(24,22,text="LIVE",anchor="w",fill="#67e8f9",font=("Tahoma",10,"bold"))
        c.create_text(76,22,text=f"{selected.get('symbol','—')}  •  {'LONG' if str(selected.get('side'))=='long' else 'SHORT'}",anchor="w",fill="#f8fafc",font=("Tahoma",13,"bold"))
        pnl=safe_float(selected.get("live_pct")); pnlc="#22c55e" if pnl>=0 else "#fb7185"
        c.create_text(right,22,text=f"P/L {pnl:+.3f}%",anchor="e",fill=pnlc,font=("Tahoma",13,"bold"))
        # Plot frame + subtle grid.
        c.create_rectangle(left,top,right,bottom,fill="#081321",outline="#17304b",width=1)
        def py(v): return bottom-(v-lo)/(hi-lo)*(bottom-top)
        def px(i): return left+i/max(1,len(hist)-1)*(right-left)
        for gy in range(6):
            y=top+gy*(bottom-top)/5; c.create_line(left,y,right,y,fill="#102338",width=1); val=hi-gy*(hi-lo)/5; c.create_text(left-8,y,text=format_price(val),anchor="e",fill="#4b6682",font=("Tahoma",7))
        for gx in range(7):
            x=left+gx*(right-left)/6; c.create_line(x,top,x,bottom,fill="#0d1d2f",width=1)
        pts=[(px(i),py(safe_float(x.get("price")))) for i,x in enumerate(hist)]
        # glow layers
        if len(pts)>1:
            c.create_line(*sum(([x,y] for x,y in pts),[]),fill="#123b55",width=7,smooth=True)
            c.create_line(*sum(([x,y] for x,y in pts),[]),fill="#38bdf8",width=2,smooth=True)
        # Current pulse.
        cx,cy=pts[-1];
        for rr,outline in ((10,"#0c4a6e"),(7,"#075985"),(4,"#67e8f9")):
            c.create_oval(cx-rr,cy-rr,cx+rr,cy+rr,fill="#081321",outline=outline,width=1)
        for name,v,color in levels:
            if v<=0:continue
            y=py(v); c.create_line(left,y,right,y,fill=color,width=1,dash=(7,4)); c.create_text(right-5,y-9,text=f"{name}  {format_price(v)}",anchor="e",fill=color,font=("Tahoma",8,"bold"))
        # Right glass panel.
        px0=right+16; c.create_rectangle(px0,top,right+225,bottom,fill="#081321",outline="#17304b")
        stage=self._trade_stage(selected); c.create_text(px0+14,top+18,text="POSITION",anchor="w",fill="#64748b",font=("Tahoma",8,"bold")); c.create_text(px0+14,top+40,text=stage,anchor="w",fill="#f8fafc",font=("Tahoma",10,"bold"))
        metrics=[("Entry",entry,"#f59e0b"),("Live",cur,"#67e8f9"),("TP1",levels[1][1],"#22c55e"),("TP2",levels[2][1],"#86efac"),("SL",levels[3][1],"#ef4444")]
        yy=top+76
        for name,val,color in metrics:
            if val<=0:continue
            c.create_rectangle(px0+10,yy,right+214,yy+43,fill="#0c1828",outline="#13283d"); c.create_rectangle(px0+10,yy,px0+13,yy+43,fill=color,outline=""); c.create_text(px0+22,yy+13,text=name,anchor="w",fill="#94a3b8",font=("Tahoma",8,"bold")); c.create_text(right+205,yy+13,text=format_price(val),anchor="e",fill="#f8fafc",font=("Tahoma",9,"bold")); yy+=51
        c.create_text(left,bottom+25,text=f"{len(hist)} price points  •  live refresh  •  risk levels",anchor="w",fill="#38516b",font=("Tahoma",8))
        self.tm_chart_title.config(text=f"{selected.get('symbol','—')}  |  {stage}  |  P/L {pnl:+.3f}%")
    except Exception as exc:
        logging.debug("V13 chart failed: %s",exc)


def _v13_build_ui_wrapper(self):
    _legacy_build_ui_for_unified(self)
    try:
        _unified_build_tab(self)
        # _make_scrollable_tab returns the inner scrollable frame; ttk.Notebook
        # manages its outer host frame. Selecting the inner canvas frame raises
        # TclError: "... is not managed by .!notebook".
        _tab_host = getattr(self.unified_center_tab, "_scroll_outer", None)
        self.notebook.select(_tab_host if _tab_host is not None else self.unified_center_tab)
        # Auto-start the center after the GUI has painted. The work is fully
        # threaded/resumable; this is not a blocking startup task.
        self.root.after(V13_CENTER_AUTO_DELAY_MS, lambda: self._unified_startup_check())
        self.root.after(V13_CENTER_AUTO_DELAY_MS+200, lambda: _v13_start_quote_loop(self))
    except Exception as exc:
        logging.exception("V13 unified UI build failed: %s",exc)


def _v13_startup_check(self):
    self._unified_ui_log('V13: مرکز هوش بازار فعال شد؛ صف خودکار به ترتیب 1m → 5m → 15m → 30m → 1h → 4h → 1d شروع می‌شود.')
    try:self.unified_status.config(text='در حال آماده‌سازی صف 1m ...',fg="#22c55e")
    except Exception:pass
    if V13_AUTO_START_UNIFIED:
        try:
            self.root.after(600, lambda: self._unified_start_update(False))
        except Exception:pass


def _v13_start(self):
    # Call original start, then attach the independent quote loop.
    _v13_original_start(self)
    try:_v13_start_quote_loop(self)
    except Exception:pass


def _v13_stop(self):
    self._v13_quote_stop=True
    try:_v13_original_stop(self)
    except Exception:pass


def _v13_scan_smart_market(self):
    """Run Smart Execution and allow both normal green and strong signals into the trade gate."""
    def worker():
        try:
            self.set_tab_activity("🧠 اجرای هوشمند", "در حال اسکن", "active")
            universe=get_smart_trade_universe(MAX_SMART_SCAN_SYMBOLS); rows=[]
            for sym in universe:
                try:
                    r=unified_ai_decision(sym)
                    if not r or not r.get("price") or (not r.get("history_ok") and not r.get("rsi")):
                        r=self._fallback_ai_row(sym)
                    if r:
                        rows.append(r); self._ensure_signal_from_decision(r,source="اجرای هوشمند")
                except Exception as exc:
                    logging.debug("V13 smart scan %s: %s",sym,exc)
            rows.sort(key=lambda x:abs(float(x.get("score",50))-50)+float(x.get("confidence",0))*.15,reverse=True); rows=rows[:100]
            if self._auto_trade_state:
                for r in rows:
                    d=str(r.get("decision","WAIT")); conf=safe_float(r.get("confidence")); rr=safe_float(r.get("risk_reward")); score=safe_float(r.get("score"))
                    strong=d in ("BUY++","SELL++")
                    normal=d in ("BUY","SELL")
                    if strong:
                        eligible=conf>=V13_STRONG_AUTO_TRADE_CONF and rr>=V13_STRONG_AUTO_TRADE_RR
                    else:
                        eligible=normal and conf>=V13_NORMAL_AUTO_TRADE_CONF and rr>=V13_NORMAL_AUTO_TRADE_RR and abs(score-50)>=12
                    if eligible:
                        try:
                            place_order(r["symbol"],"long" if d.startswith("BUY") else "short",r.get("trade_plan"),score,conf)
                            logging.info("V13 auto-trade eligible %s %s conf=%.1f rr=%.2f",r.get("symbol"),d,conf,rr)
                        except Exception as exc:
                            logging.debug("V13 auto trade %s: %s",r.get("symbol"),exc)
            self.root.after(0,lambda rs=rows:self._show_smart_rows(rs))
        finally:
            self.root.after(0,lambda:self.smart_status.config(text="اسکن تکمیل شد"))
    self.smart_status.config(text="در حال تحلیل...")
    threading.Thread(target=worker,daemon=True,name="V13SmartExecution").start()


# Install V13 overrides before application construction.
try:
    _v13_original_start=NobitexScalperPro.start
    _v13_original_stop=NobitexScalperPro.stop
    NobitexScalperPro.start=_v13_start
    NobitexScalperPro.stop=_v13_stop
    NobitexScalperPro.scan_smart_market=_v13_scan_smart_market
    NobitexScalperPro._build_layout_settings_tab=_v13_clean_layout_build
    NobitexScalperPro._apply_saved_layout_settings=_v13_clean_apply_saved_layout
    NobitexScalperPro._reset_layout_settings=_v13_reset_layout
    NobitexScalperPro._draw_trade_chart=_v13_pretty_trade_chart
    NobitexScalperPro._unified_startup_check=_v13_startup_check
    NobitexScalperPro._build_ui=_v13_build_ui_wrapper
except Exception as exc:
    logging.exception("V13 override install failed: %s",exc)



# ============================================================
# V18 DIAGNOSTIC / RISK + TRADE-CLOSE TELEMETRY PATCH
# Based on observed diagnostic run 2026-09-29:
# - legacy confirmation produced 0 confirmed decisions
# - scan cycles produced signals but no decision/gate/trade telemetry
# - cycle latency was highly variable
# This patch adds auditable per-symbol decision telemetry and a single
# auto-trade eligibility gate for BOTH normal (green) and strong (gold) signals.
# It does not claim that a signal is profitable or guarantee execution.
# ============================================================

V17_DIAG_DIR = Path(APP_DATA_DIR) / "diagnostics"
V17_DIAG_DIR.mkdir(parents=True, exist_ok=True)
V17_LOCAL_DIAG_DIR = Path(__file__).resolve().parent / "logs"
try:
    V17_LOCAL_DIAG_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
V17_SESSION = datetime.now().strftime('%Y%m%d_%H%M%S')
V17_DIAG_FILE = V17_DIAG_DIR / f"decision_trace_{V17_SESSION}.jsonl"
V17_LOCAL_DIAG_FILE = V17_LOCAL_DIAG_DIR / f"decision_trace_{V17_SESSION}.jsonl"
_v17_diag_lock = threading.Lock()
_v17_decision_counter = 0


def _v17_json_safe(value, depth=0):
    if depth > 3:
        return str(value)[:300]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _v17_json_safe(v, depth + 1) for k, v in list(value.items())[:120]}
    if isinstance(value, (list, tuple, set)):
        return [_v17_json_safe(v, depth + 1) for v in list(value)[:80]]
    return str(value)[:500]


def _v17_log(event, **data):
    rec = {
        'ts': time.time(),
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'session': V17_SESSION,
        'event': event,
        'data': _v17_json_safe(data),
    }
    line = json.dumps(rec, ensure_ascii=False, separators=(',', ':')) + '\n'
    errors = []
    for target in (V17_LOCAL_DIAG_FILE, V17_DIAG_FILE):
        try:
            with _v17_diag_lock:
                with open(target, 'a', encoding='utf-8') as f:
                    f.write(line)
                    f.flush()
            return True
        except Exception as exc:
            errors.append(f'{target}: {exc!r}')
    # Never hide a diagnostic-write failure. It is itself a diagnostic event.
    try:
        logging.error('[V17-DIAG-WRITE-FAILED] event=%s errors=%s', event, ' | '.join(errors))
    except Exception:
        pass
    return False


def _v17_num(v, default=0.0):
    try:
        x = float(v)
        return default if math.isnan(x) or math.isinf(x) else x
    except Exception:
        return default


def _v17_snapshot(r):
    if not isinstance(r, dict):
        return {'raw_type': type(r).__name__}
    keys = (
        'symbol','decision','signal_state','score','confidence','quality','model_quality',
        'talaye_score','talaye_quality','mtf_score','mtf','smc_score','discovery_score',
        'discovery_direction','buy_probability','sell_probability','adx','volume_ratio',
        'risk_reward','entry','stop_loss','tp1','tp2','regime','history_ok','data_status',
        'choch_bullish','choch_bearish','indicator_confluence','price'
    )
    return {k: _v17_json_safe(r.get(k)) for k in keys if k in r}


def _v17_decision_class(r):
    d = str((r or {}).get('decision', 'WAIT')).upper()
    if d in ('BUY++','SELL++'):
        return 'GOLD'
    if d in ('BUY','SELL'):
        return 'GREEN'
    return 'WAIT'


def _v17_trade_gate(r):
    """One deterministic gate used for normal and strong signals.
    Green and gold are both eligible; gold only gets stricter thresholds.
    """
    if not isinstance(r, dict):
        return False, 'invalid_decision'
    d = str(r.get('decision', 'WAIT')).upper()
    if d not in ('BUY','SELL','BUY++','SELL++'):
        return False, 'decision_is_wait'
    if not r.get('history_ok', True) and not r.get('rsi'):
        return False, 'history_not_ready'
    conf = _v17_num(r.get('confidence'), 0)
    rr = _v17_num(r.get('risk_reward'), 0)
    score = _v17_num(r.get('score'), 50)
    cls = _v17_decision_class(r)
    # Keep the existing V13 thresholds, but apply them through one gate.
    if cls == 'GOLD':
        min_conf = _v17_num(globals().get('V13_STRONG_AUTO_TRADE_CONF'), 82)
        min_rr = _v17_num(globals().get('V13_STRONG_AUTO_TRADE_RR'), 1.8)
        if conf < min_conf:
            return False, f'gold_confidence<{min_conf:g}'
        if rr < min_rr:
            return False, f'gold_rr<{min_rr:g}'
    else:
        min_conf = max(78.0, _v17_num(globals().get('V13_NORMAL_AUTO_TRADE_CONF'), 72))
        min_rr = max(1.8, _v17_num(globals().get('V13_NORMAL_AUTO_TRADE_RR'), 1.5))
        if conf < min_conf:
            return False, f'green_confidence<{min_conf:g}'
        if rr < min_rr:
            return False, f'green_rr<{min_rr:g}'
        if abs(score - 50.0) < 12.0:
            return False, 'green_score_distance<12'
    return True, 'PASS'


# Preserve the currently-installed (latest) decision implementation.
try:
    _v17_original_unified_ai_decision = unified_ai_decision
except Exception:
    _v17_original_unified_ai_decision = None


def unified_ai_decision(symbol):
    global _v17_decision_counter
    _v17_decision_counter += 1
    did = f'D{int(time.time()*1000)}-{_v17_decision_counter}'
    t0 = time.perf_counter()
    try:
        if _v17_original_unified_ai_decision is None:
            raise RuntimeError('original unified_ai_decision unavailable')
        r = _v17_original_unified_ai_decision(symbol)
        elapsed = (time.perf_counter() - t0) * 1000.0
        snap = _v17_snapshot(r)
        eligible, reason = _v17_trade_gate(r)
        snap['decision_id'] = did
        snap['trade_class'] = _v17_decision_class(r)
        snap['trade_gate'] = eligible
        snap['trade_gate_reason'] = reason
        snap['elapsed_ms'] = round(elapsed, 2)
        _v17_log('SIGNAL_DECISION', **snap)
        return r
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        _v17_log('SIGNAL_DECISION_ERROR', symbol=str(symbol), decision_id=did,
                 elapsed_ms=round(elapsed,2), error=repr(exc))
        raise


# Instrument the simulation order function without changing its risk mechanics.
try:
    _v17_original_place_order = place_order
except Exception:
    _v17_original_place_order = None


def place_order(sym, direction, plan=None, score=0.0, confidence=0.0):
    before = len(auto_trade_log)
    t0 = time.perf_counter()
    _v17_log('TRADE_ATTEMPT', symbol=canonical(sym), direction=direction,
              score=_v17_num(score), confidence=_v17_num(confidence), mode=('real_margin' if _real_trading_is_enabled() else 'simulation'))
    try:
        if _v17_original_place_order is None:
            return None
        result = _v17_original_place_order(sym, direction, plan, score, confidence)
        after = len(auto_trade_log)
        _v17_log('TRADE_RESULT', symbol=canonical(sym), direction=direction,
                  accepted=after > before, elapsed_ms=round((time.perf_counter()-t0)*1000,2),
                  open_trade_count=after)
        return result
    except Exception as exc:
        _v17_log('TRADE_RESULT', symbol=canonical(sym), direction=direction,
                  accepted=False, elapsed_ms=round((time.perf_counter()-t0)*1000,2), error=repr(exc))
        raise


# Replace the V13 smart scan with a bounded, fully logged version. Both GREEN
# and GOLD decisions use the same final gate; GOLD is not the only tradable class.
def _v17_scan_smart_market(self):
    def worker():
        started = time.perf_counter()
        rows = []
        universe = get_smart_trade_universe(MAX_SMART_SCAN_SYMBOLS)
        _v17_log('SCAN_CYCLE_START', count=len(universe), symbols=list(universe)[:100])
        counts = {'WAIT':0,'GREEN':0,'GOLD':0,'GATE_PASS':0,'GATE_FAIL':0,'TRADE':0}
        for sym in universe:
            t0 = time.perf_counter()
            try:
                r = unified_ai_decision(sym)
                if not r or (not r.get('price') and not r.get('history_ok')):
                    r = self._fallback_ai_row(sym)
                rows.append(r)
                cls = _v17_decision_class(r)
                counts[cls] = counts.get(cls,0) + 1
                eligible, reason = _v17_trade_gate(r)
                if eligible:
                    counts['GATE_PASS'] += 1
                else:
                    counts['GATE_FAIL'] += 1
                _v17_log('TRADE_GATE', symbol=sym, decision=r.get('decision','WAIT'),
                         signal_class=cls, passed=eligible, reason=reason,
                         confidence=_v17_num(r.get('confidence')), rr=_v17_num(r.get('risk_reward')),
                         score=_v17_num(r.get('score')), elapsed_ms=round((time.perf_counter()-t0)*1000,2))
                try:
                    self._ensure_signal_from_decision(r, source='V17 unified smart scan')
                except Exception as exc:
                    _v17_log('SIGNAL_BRIDGE_ERROR', symbol=sym, error=repr(exc))
                if self._auto_trade_state and eligible:
                    try:
                        direction = 'long' if str(r.get('decision','')).startswith('BUY') else 'short'
                        before = len(auto_trade_log)
                        place_order(sym, direction, r.get('trade_plan'), r.get('score',0), r.get('confidence',0))
                        if len(auto_trade_log) > before:
                            counts['TRADE'] += 1
                    except Exception as exc:
                        _v17_log('AUTO_TRADE_ERROR', symbol=sym, error=repr(exc))
            except Exception as exc:
                _v17_log('SIGNAL_ERROR', symbol=sym, error=repr(exc), elapsed_ms=round((time.perf_counter()-t0)*1000,2))
        rows.sort(key=lambda x: (_v17_num(x.get('confidence')), abs(_v17_num(x.get('score'),50)-50)), reverse=True)
        rows = rows[:100]
        elapsed = time.perf_counter()-started
        _v17_log('SCAN_CYCLE_END', completed=len(universe), elapsed_sec=round(elapsed,3), counts=counts)
        try:
            self.root.after(0, lambda rs=rows:self._show_smart_rows(rs))
            self.root.after(0, lambda c=counts,e=elapsed:self.smart_status.config(
                text=f"اسکن: {c.get('GREEN',0)} سبز | {c.get('GOLD',0)} طلایی | Gate={c.get('GATE_PASS',0)} | {e:.1f}s"))
        except Exception:
            pass
    self.smart_status.config(text='در حال تحلیل و ثبت تصمیم‌ها...')
    threading.Thread(target=worker, daemon=True, name='V17UnifiedSmartExecution').start()

try:
    NobitexScalperPro.scan_smart_market = _v17_scan_smart_market
except Exception:
    pass

# Overview bootstrap previously wrote a permanent 0.00% change. Keep the row
# shape but compute change from cached 24h stats whenever available.
def _v17_refresh_overview_prices(self):
    try:
        existing = {str(self.main_table.item(i).get('values',[None])[0]): i for i in self.main_table.get_children()}
        for sym in list(SYMBOLS):
            key = canonical(sym)
            price = valid_symbols_map.get(key)
            if price is None:
                continue
            change = None
            meta = globals().get('MARKET_STATS_META', {})
            if isinstance(meta, dict):
                m = meta.get(key) or meta.get(sym) or {}
                if isinstance(m, dict):
                    for k in ('change24h','change_24h','change_pct','percent_change','percentage'):
                        if m.get(k) not in (None,''):
                            change = _v17_num(m.get(k), None); break
            if change is None:
                # Preserve unknown rather than displaying a false 0.00%.
                change_display = '—'
            else:
                change_display = f'{change:+.2f}%'
            vals = (key, format_price(price), change_display, '—','—','—','—','—','—','—',
                    'در حال تحلیل','نگهداری',0,'—','—','—','—','neutral','—','')
            if key in existing:
                iid = existing[key]
                cur = list(self.main_table.item(iid).get('values',()))
                if len(cur) == 20:
                    cur[1] = vals[1]; cur[2] = vals[2]
                    self.main_table.item(iid, values=tuple(cur))
            else:
                self.main_table.insert('', 'end', values=vals, tags=('neutral',))
    except Exception as exc:
        _v17_log('OVERVIEW_REFRESH_ERROR', error=repr(exc))

try:
    NobitexScalperPro.refresh_overview_prices = _v17_refresh_overview_prices
except Exception:
    pass

_v17_log('STARTUP', python_version=sys.version.split()[0], diagnostic_file=str(V17_DIAG_FILE),
         local_diagnostic_file=str(V17_LOCAL_DIAG_FILE), ml_available=_ML_AVAILABLE)

# ============================================================
# V22 FINAL INTEGRATION + UI POLISH
# ============================================================
V22_VERSION = "V22_UNIFIED_RISK_UI"
MAX_MARGIN_LEVERAGE = 5.0


def _v22_apply_canonical_plan_to_decision(r):
    """Attach one canonical leverage/TP/SL plan to every executable decision."""
    if not isinstance(r, dict):
        return r
    try:
        decision = str(r.get("decision", "WAIT"))
        entry = safe_float(r.get("price") or r.get("entry"), 0.0)
        confidence = safe_float(r.get("confidence"), 0.0)
        plan = r.get("trade_plan") if isinstance(r.get("trade_plan"), dict) else None
        if decision.startswith(("BUY", "SELL")) and entry > 0:
            # Reuse historical/current dataframe if available; otherwise retain the
            # existing structural SL and re-price TP from the canonical profile.
            df = None
            try:
                df, _ = smart_ai_history(canonical(r.get("symbol", "")), 15, bars=220)
            except Exception:
                pass
            if df is not None and not df.empty:
                plan = _v22_build_trade_plan(df, decision, entry, confidence)
            elif plan:
                lev = _v22_confidence_to_leverage(confidence)
                p1, p2 = _v22_tp_profile(lev)
                side = "long" if decision.startswith("BUY") else "short"
                plan = dict(plan)
                plan["leverage"] = lev
                plan["confidence"] = confidence
                plan["tp1_price_pct"] = p1
                plan["tp2_price_pct"] = p2
                plan["tp1_leveraged_return_pct"] = p1 * lev
                plan["tp2_leveraged_return_pct"] = p2 * lev
                plan["tp1"] = entry*(1+p1/100) if side == "long" else entry*(1-p1/100)
                plan["tp2"] = entry*(1+p2/100) if side == "long" else entry*(1-p2/100)
                plan["rr"] = abs(plan["tp2"]-entry)/max(abs(entry-safe_float(plan.get("sl"),entry)),1e-12)
            if plan:
                r["trade_plan"] = plan
                r["entry"] = plan.get("entry", entry)
                r["stop_loss"] = plan.get("sl")
                r["tp1"] = plan.get("tp1")
                r["tp2"] = plan.get("tp2")
                r["risk_reward"] = plan.get("rr", r.get("risk_reward", 0.0))
                r["leverage"] = plan.get("leverage", _v22_confidence_to_leverage(confidence))
                r["tp1_price_pct"] = plan.get("tp1_price_pct")
                r["tp2_price_pct"] = plan.get("tp2_price_pct")
                r["tp1_leveraged_return_pct"] = plan.get("tp1_leveraged_return_pct")
                r["tp2_leveraged_return_pct"] = plan.get("tp2_leveraged_return_pct")
        else:
            r.setdefault("leverage", 1.0)
        return r
    except Exception as exc:
        logging.debug("V22 canonical plan failed: %s", exc)
        return r


# Preserve the diagnostic wrapper while making the canonical plan the final output.
try:
    _v22_original_decision_wrapper = unified_ai_decision
except Exception:
    _v22_original_decision_wrapper = None


def unified_ai_decision(symbol):
    if _v22_original_decision_wrapper is None:
        return {"symbol": canonical(symbol), "decision": "WAIT", "confidence": 0.0}
    r = _v22_original_decision_wrapper(symbol)
    return _v22_apply_canonical_plan_to_decision(r)


# ---------------- V22 UI polish ----------------
def _v22_style_ui(root):
    try:
        root.title("Nobitex AI Scalper Pro V22 — Unified Risk • TP/SL • Profit Lock")
        root.minsize(1180, 760)
        root.geometry("1600x950")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("V22.TNotebook", background="#07111f", borderwidth=0)
        style.configure("V22.TNotebook.Tab", background="#172033", foreground="#cbd5e1",
                        padding=(14, 8), font=("Tahoma", 9, "bold"), borderwidth=0)
        style.map("V22.TNotebook.Tab",
                  background=[("selected", "#2563eb"), ("active", "#334155")],
                  foreground=[("selected", "#ffffff"), ("active", "#ffffff")])
        style.configure("V22.Vertical.TScrollbar", background="#1e293b", troughcolor="#0b1220",
                        arrowcolor="#94a3b8", borderwidth=0)
        style.configure("V22.Horizontal.TScrollbar", background="#1e293b", troughcolor="#0b1220",
                        arrowcolor="#94a3b8", borderwidth=0)
        try:
            root.option_add("*Font", "Tahoma 9")
            root.option_add("*Button.Cursor", "hand2")
        except Exception:
            pass

        # A compact status banner makes the active risk mode obvious.
        banner = tk.Frame(root, bg="#081321", bd=1, relief="solid",
                          highlightthickness=1, highlightbackground="#1e3a5f")
        banner.pack(fill="x", padx=8, pady=(5, 3), before=getattr(root, "notebook", root))
        tk.Label(banner, text="⚡ V22  مرکز معاملات یکپارچه", bg="#081321", fg="#67e8f9",
                 font=("Tahoma", 12, "bold")).pack(side="right", padx=12, pady=6)
        tk.Label(banner, text="Confidence → Leverage → SL/TP → Profit Lock",
                 bg="#081321", fg="#94a3b8", font=("Tahoma", 9)).pack(side="right", padx=8)
        tk.Label(banner, text="1x  2x  3x  4x  5x",
                 bg="#0f2740", fg="#93c5fd", font=("Tahoma", 9, "bold"),
                 padx=10, pady=3).pack(side="left", padx=10)
    except Exception as exc:
        logging.debug("V22 style banner: %s", exc)


def _v22_reflow_top_controls(self):
    """Prevent the wide legacy control row from pushing buttons off-screen."""
    try:
        root = self.root
        ctrl = None
        for w in root.winfo_children():
            if isinstance(w, tk.Frame) and str(w.cget("bg")) in ("#222", "#222222"):
                if any(isinstance(c, tk.Button) for c in w.winfo_children()):
                    ctrl = w
                    break
        if ctrl is None:
            return
        ctrl.configure(height=150, bd=0, highlightthickness=0)
        # Find the large settings frame containing the five scale blocks.
        settings = None
        expected_blocks = [getattr(self, "_scale_blocks", {}).get(k) for k in
                           ("trailing_activate", "trailing_stage2", "trailing_stage1_retrace",
                            "trailing_stage2_retrace", "stop_loss")]
        expected_blocks = [b for b in expected_blocks if b is not None]
        for child in ctrl.winfo_children():
            try:
                if expected_blocks and all(getattr(b, "master", None) is child for b in expected_blocks):
                    settings = child
                    break
            except Exception:
                pass
        if settings is None:
            settings = next((c for c in ctrl.winfo_children()
                             if isinstance(c, tk.Frame) and len(c.winfo_children()) >= 5), None)
        if settings is not None:
            try: settings.pack_forget()
            except Exception: pass
            settings.pack(side="top", fill="x", padx=8, pady=(5, 2), before=ctrl.winfo_children()[0])
            for col in range(6):
                settings.columnconfigure(col, weight=1)
            # Put five large controls into two responsive rows.
            ordered = ["trailing_activate", "trailing_stage2", "trailing_stage1_retrace",
                       "trailing_stage2_retrace", "stop_loss"]
            for i, key in enumerate(ordered):
                block = getattr(self, "_scale_blocks", {}).get(key)
                if block is None: continue
                try:
                    block.grid_forget()
                    block.grid(row=i//3, column=i%3, sticky="ew", padx=4, pady=3)
                    if hasattr(block, "scale"):
                        block.scale.configure(length=95)
                except Exception:
                    pass
            try:
                settings.rowconfigure(0, weight=1); settings.rowconfigure(1, weight=1)
            except Exception: pass
        # Remove the duplicate tiny manual Talaye controls from the crowded first row.
        for child in list(ctrl.winfo_children()):
            try:
                txt = child.cget("text") if "text" in child.keys() else ""
            except Exception: txt = ""
            if "طلایه آستانه" in str(txt):
                try: child.pack_forget()
                except Exception: pass
        # The quick-risk frame remains visible but can wrap naturally below the buttons.
        try:
            for child in ctrl.winfo_children():
                if isinstance(child, tk.LabelFrame) and "کنترل سریع ریسک" in str(child.cget("text")):
                    child.pack(side="right", padx=8, pady=3)
        except Exception: pass
        ctrl.update_idletasks()
    except Exception as exc:
        logging.debug("V22 control reflow failed: %s", exc)


def _v22_polish_notebook(self):
    try:
        nb = self.notebook
        style = ttk.Style(self.root)
        style.configure("TNotebook", background="#07111f", borderwidth=0, tabmargins=(3,3,3,0))
        style.configure("TNotebook.Tab", background="#172033", foreground="#cbd5e1",
                        padding=(13,7), font=("Tahoma",9,"bold"), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected","#2563eb"),("active","#334155")],
                  foreground=[("selected","white"),("active","white")])
        # Give every page a little breathing room and ensure the page canvas follows the viewport.
        for tab_id in nb.tabs():
            try:
                host = nb.nametowidget(tab_id)
                canvas = getattr(host, "_scroll_canvas", None)
                if canvas:
                    canvas.configure(bg="#0b1220", highlightthickness=0)
                inner = getattr(host, "_scroll_inner", None)
                if inner:
                    inner.configure(bg="#0b1220")
            except Exception:
                pass
    except Exception as exc:
        logging.debug("V22 notebook polish failed: %s", exc)


try:
    _v22_original_build_ui = NobitexScalperPro._build_ui
except Exception:
    _v22_original_build_ui = None


def _v22_build_ui(self):
    if _v22_original_build_ui is not None:
        _v22_original_build_ui(self)
    try:
        _v22_style_ui(self.root)
        _v22_reflow_top_controls(self)
        _v22_polish_notebook(self)
        # The monitor is the operational landing page, but now all tabs remain reachable.
        try:
            self.notebook.select(self.trade_monitor_tab)
        except Exception:
            pass
        logging.info("V22 UI polish active")
    except Exception as exc:
        logging.exception("V22 UI polish failed: %s", exc)


NobitexScalperPro._build_ui = _v22_build_ui

# Persist the V22 version in the diagnostic stream.
try:
    _v17_log('V22_STARTUP', version=V22_VERSION, max_leverage=MAX_MARGIN_LEVERAGE,
             tp_profile={str(k): v for k,v in {1:(1,2),2:(1,2),3:(.9,1.8),4:(.8,1.6),5:(.7,1.4)}.items()})
except Exception:
    pass


# ============================================================
# V22.4 STABLE UI
# One UI layer only.  No recursive canvas Configure handlers,
# no repeated pack/grid reflow passes, and no automatic heavy ML/history work.
# ============================================================
V22_4_VERSION = "V22.4_STABLE_UI"
V13_AUTO_START_UNIFIED = False


def _v224_find_legacy_ctrl(self):
    root = self.root
    for w in root.winfo_children():
        try:
            if isinstance(w, tk.Frame) and str(w.cget("bg")) in ("#222", "#222222"):
                if any(isinstance(c, tk.Button) for c in w.winfo_children()):
                    return w
        except Exception:
            pass
    return None


def _v224_arrange_toolbar(self):
    """Arrange the existing toolbar exactly once using grid; no after/resize loops."""
    ctrl = _v224_find_legacy_ctrl(self)
    if ctrl is None:
        return
    try:
        ctrl.configure(bg="#0b1220", bd=0, highlightthickness=1,
                       highlightbackground="#1e3a5f", padx=8, pady=7)
        ctrl.pack_configure(fill="x", padx=8, pady=(5, 4))

        buttons=[]; quick=None; settings=None
        for w in list(ctrl.winfo_children()):
            try:
                if isinstance(w, tk.Button):
                    txt=str(w.cget("text")).strip()
                    if txt not in ("اعمال", "Apply"):
                        buttons.append(w)
                    else:
                        w.pack_forget(); w.grid_forget(); w.place_forget(); w.lower()
                elif isinstance(w, tk.LabelFrame) and "کنترل سریع ریسک" in str(w.cget("text")):
                    quick=w
                elif w is getattr(self, "_scale_blocks", {}).get("trailing_activate", None).master if getattr(self, "_scale_blocks", {}).get("trailing_activate", None) else False:
                    settings=w
            except Exception:
                pass
        if settings is None:
            b=getattr(self, "_scale_blocks", {}).get("trailing_activate")
            settings=getattr(b, "master", None) if b else None

        # Remove legacy direct Talaye controls; the compact control remains inside settings.
        for w in list(ctrl.winfo_children()):
            try:
                txt=str(w.cget("text")) if "text" in w.keys() else ""
                if "طلایه آستانه" in txt or w is getattr(self,"talaye_value_entry",None):
                    w.pack_forget(); w.grid_forget(); w.place_forget(); w.lower()
            except Exception:
                pass

        for w in list(ctrl.winfo_children()):
            try: w.pack_forget(); w.grid_forget()
            except Exception: pass

        for c in range(8): ctrl.columnconfigure(c, weight=1, uniform="toolbar")
        ctrl.rowconfigure(0, weight=0); ctrl.rowconfigure(1, weight=0)

        if settings is not None:
            settings.configure(bg="#0b1220", bd=0)
            for w in list(settings.winfo_children()):
                try: w.grid_forget(); w.pack_forget()
                except Exception: pass
            order=["trailing_activate","trailing_stage2","trailing_stage1_retrace","trailing_stage2_retrace","stop_loss"]
            for i,key in enumerate(order):
                b=getattr(self,"_scale_blocks",{}).get(key)
                if b is not None:
                    b.configure(bg="#111c2d", fg="#93c5fd")
                    b.grid(row=i//3,column=i%3,sticky="ew",padx=3,pady=3)
                    try: b.scale.configure(length=105, highlightthickness=0)
                    except Exception: pass
            for c in range(3): settings.columnconfigure(c,weight=1,uniform="risk")
            for w in list(settings.winfo_children()):
                try:
                    txt=str(w.cget("text")) if "text" in w.keys() else ""
                    if "قانون:" in txt:
                        w.grid(row=2,column=0,columnspan=3,sticky="ew",padx=6,pady=(2,3))
                    elif "فعال‌سازی تریلینگ" in txt:
                        w.grid(row=3,column=0,sticky="w",padx=6,pady=(2,4))
                    elif isinstance(w,tk.Frame) and any(isinstance(c,tk.Spinbox) for c in w.winfo_children()):
                        w.grid(row=3,column=1,columnspan=2,sticky="w",padx=6,pady=(2,4))
                except Exception: pass
            settings.grid(row=0,column=0,columnspan=8,sticky="ew",padx=0,pady=(0,5))

        # Seven main actions + quick risk card. All have equal cells and remain inside the window.
        for i,btn in enumerate(buttons[:7]):
            try:
                btn.configure(bg="#0891b2", activebackground="#06b6d4", fg="white",
                              relief="flat", bd=0, highlightthickness=1,
                              highlightbackground="#164e63", font=("Tahoma",9,"bold"),
                              padx=4,pady=5,width=11,cursor="hand2")
                btn.grid(row=1,column=i,sticky="ew",padx=3,pady=(0,2))
            except Exception: pass
        if quick is not None:
            quick.configure(bg="#172033", fg="#93c5fd", bd=1, relief="solid", padx=5,pady=2)
            quick.grid(row=1,column=7,sticky="ew",padx=3,pady=(0,2))
        ctrl.update_idletasks()
    except Exception as exc:
        logging.debug("V22.4 toolbar arrange failed: %s", exc)


def _v224_rounded_tabbar(self):
    """Stable rounded tab navigation. Draws only on tab changes/initialization."""
    nb=self.notebook; root=self.root
    style=ttk.Style(root)
    try:
        style.layout("V224.HiddenNotebook.Tab", [])
    except Exception:
        pass
    style.configure("V224.HiddenNotebook", background="#07111f", borderwidth=0, tabmargins=0)
    nb.configure(style="V224.HiddenNotebook")

    host=tk.Frame(root,bg="#07111f",bd=0)
    host.pack(fill="x",padx=8,pady=(2,4),before=nb)
    self._v224_tab_host=host
    canvas=tk.Canvas(host,height=54,bg="#07111f",highlightthickness=0,bd=0)
    hbar=ttk.Scrollbar(host,orient="horizontal",command=canvas.xview,style="V22.Horizontal.TScrollbar")
    canvas.configure(xscrollcommand=hbar.set)
    canvas.pack(fill="x",expand=True)
    hbar.pack(fill="x")
    self._v224_tab_canvas=canvas
    self._v224_tab_items={}
    palette=["#2563eb","#0891b2","#7c3aed","#059669","#d97706","#dc2626","#db2777","#475569","#0f766e","#9333ea","#0369a1","#b45309"]

    def rounded(c,x1,y1,x2,y2,r,fill,outline=""):
        pts=[x1+r,y1,x2-r,y1,x2,y1+r,x2,y2-r,x2-r,y2,x1+r,y2,x1,y2-r,x1,y1+r]
        return c.create_polygon(pts, smooth=True, splinesteps=12, fill=fill, outline=outline)

    def redraw(*_):
        try:
            canvas.delete("all"); self._v224_tab_items={}
            x=8; y1=7; y2=47
            for i,tid in enumerate(nb.tabs()):
                label=str(nb.tab(tid,"text"))
                fill=palette[i%len(palette)]
                selected=(tid==nb.select())
                if selected:
                    fill="#0ea5e9"
                width=max(112,min(190,24+len(label)*9))
                tag=f"tab_{i}"
                rounded(canvas,x,y1,x+width,y2,14,fill,"#7dd3fc" if selected else "#243b53")
                canvas.create_text(x+width/2,(y1+y2)/2,text=label,fill="white",
                                   font=("Tahoma",9,"bold"),tags=tag)
                canvas.create_rectangle(x,y1,x+width,y2,outline="",fill="",tags=tag)
                def click(e,t=tid):
                    try: nb.select(t)
                    except Exception: pass
                canvas.tag_bind(tag,"<Button-1>",click)
                canvas.tag_bind(tag,"<Enter>",lambda e,t=tag: canvas.configure(cursor="hand2"))
                canvas.tag_bind(tag,"<Leave>",lambda e,t=tag: canvas.configure(cursor=""))
                self._v224_tab_items[tid]=(x,x+width)
                x+=width+8
            canvas.configure(scrollregion=(0,0,max(x,canvas.winfo_width()),54))
        except Exception as exc:
            logging.debug("V22.4 tab redraw failed: %s",exc)
    self._v224_redraw_tabs=redraw
    nb.bind("<<NotebookTabChanged>>",redraw,add="+")
    redraw()


def _v224_build_ui(self):
    # Use the stable V13/V12 construction exactly once; do not layer V22.1/2/3 UI overrides.
    if _v22_original_build_ui is not None:
        _v22_original_build_ui(self)
    else:
        _legacy_build_ui_for_unified(self)
    try:
        _v224_arrange_toolbar(self)
        _v224_rounded_tabbar(self)
        # Start on the unified trading center if available, otherwise overview.
        target=getattr(self,"trade_monitor_tab",None) or getattr(self,"trading_center_tab",None)
        if target is not None:
            try: self.notebook.select(target)
            except Exception:
                try: self.notebook.select(getattr(target,"_scroll_outer",target))
                except Exception: pass
        logging.info("V22.4 stable UI active")
    except Exception:
        logging.exception("V22.4 stable UI setup failed")

NobitexScalperPro._build_ui=_v224_build_ui

# Final entry point
if __name__ == "__main__":
    try:
        ensure_csv_header()
    except Exception:
        pass
    root=tk.Tk()
    root.configure(bg="#07111f")
    nob=NobitexScalperPro(root)
    try:
        root.mainloop()
    finally:
        try:
            if hasattr(nob,"executor") and nob.executor:
                nob.executor.shutdown(wait=False,cancel_futures=True)
        except Exception:
            pass
