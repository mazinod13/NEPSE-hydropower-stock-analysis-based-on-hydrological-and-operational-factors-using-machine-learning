"""
=============================================================================
 NEPSE Hydropower Companies — Multi-Company ML Preprocessing Pipeline
=============================================================================
 Companies : RADHI, RHPC, UPPER  (edit COMPANIES dict to match your tickers)
 Purpose   : Load → Clean → Engineer Features → Scale → Split → Visualise
             Produces thesis-quality plots and TensorFlow-ready sequences.
 Usage     : Edit the COMPANIES dict below, then run:
                 python hydropower_preprocessing.py
=============================================================================
"""

# ── standard ──────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")

# ── visualisation ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats

# ── sklearn ────────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# ── TensorFlow ──────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout,
                                     BatchNormalization, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
print(f"TensorFlow {tf.__version__} loaded ✓")

# ── optional technical indicators ─────────────────────────────────────────
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False
    print("Warning: `ta` library not found. Using manual indicator fallback.")


# =============================================================================
# 0.  CONFIGURATION  ← Edit this section to match your file structure
# =============================================================================

# Each company maps to a dict of {price_field: csv_path}
COMPANIES = {
    "BARUN": {
        "open"      : "Datasets\Price Volume\BARUN\OPEN.csv",
        "high"      : "Datasets\Price Volume\BARUN\HIGH.csv",
        "low"       : "Datasets\Price Volume\BARUN\LOW.csv",
        "close"     : "Datasets\Price Volume\BARUN\CLOSE.csv",
        "adj_close" : "Datasets\Price Volume\BARUN\ADJ CLOSE.csv",
        "volume"    : "Datasets\Price Volume\BARUN\VOLUME.csv",
    },
    "RADHI": {
        "open"      : "Datasets\Price Volume\RADHI\OPEN.csv",
        "high"      : "Datasets\Price Volume\RADHI\HIGH.csv",
        "low"       : "Datasets\Price Volume\RADHI\LOW.csv",
        "close"     : "Datasets\Price Volume\RADHI\CLOSE.csv",
        "adj_close" : "Datasets\Price Volume\RADHI\ADJ CLOSE.csv",
        "volume"    : "Datasets\Price Volume\RADHI\VOLUME.csv",
    },
    "CHCL": {
        "open"      : "Datasets\Price Volume\CHCL\OPEN.csv",
        "high"      : "Datasets\Price Volume\CHCL\HIGH.csv",
        "low"       : "Datasets\Price Volume\CHCL\LOW.csv",
        "close"     : "Datasets\Price Volume\CHCL\CLOSE.csv",
        "adj_close" : "Datasets\Price Volume\CHCL\ADJ CLOSE.csv",
        "volume"    : "Datasets\Price Volume\CHCL\VOLUME.csv",
    },
}

# External data (optional — leave as None if not available yet)
EXTERNAL_DATA = {
    "RADHI": {
        "fundamentals" : None,   # pd.DataFrame with DateTimeIndex
        "rainfall"     : None,   # pd.DataFrame with DateTimeIndex
        "production"   : None,   # pd.DataFrame with DateTimeIndex
    },
    # Repeat for other companies
}

OUTPUT_DIR         = Path("Output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_HORIZON = 5          # trading days ahead
ROLLING_WINDOWS    = [5, 10, 20, 50]
SEQ_LENGTH         = 60         # lookback window for LSTM (days)
TRAIN_RATIO        = 0.70
VAL_RATIO          = 0.15       # test = remaining 0.15
N_CV_FOLDS         = 5

# Thesis plot style
COMPANY_COLORS = {
    "RADHI" : "#1B4F72",
    "BARUN"  : "#1E8449",
    "CHCL" : "#922B21",
}
ACCENT_COLOR   = "#F39C12"

plt.rcParams.update({
    "font.family"       : "DejaVu Serif",
    "font.size"         : 15,
    "axes.titlesize"    : 16,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 15,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "figure.dpi"        : 360,
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#FAFAFA",
    "grid.alpha"        : 0.7,
    "grid.linestyle"    : "--",
    "legend.framealpha" : 0.9,
    "savefig.bbox"      : "tight",
    "savefig.dpi"       : 360,
})


# =============================================================================
# 1.  DATA LOADING
# =============================================================================

def load_company(files: dict, ticker: str) -> pd.DataFrame:
    """Load and merge all six OHLCV CSV files for one company."""

    def load_series(path, col):
        df = pd.read_csv(path, header=None, names=["date", col])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[col]    = (df[col].astype(str)
                      .str.replace(",", "", regex=False)
                      .pipe(pd.to_numeric, errors="coerce"))
        return df.dropna(subset=["date"]).set_index("date")[col]

    series = {name: load_series(path, name) for name, path in files.items()}
    df = pd.DataFrame(series).sort_index()
    print(f"  [{ticker}] Loaded {len(df)} rows  "
          f"({df.index.min().date()} → {df.index.max().date()})")
    return df


def load_all_companies(companies: dict) -> dict:
    print("\n[LOAD] Loading all companies …")
    return {ticker: load_company(files, ticker)
            for ticker, files in companies.items()}


# =============================================================================
# 2.  DATA QUALITY REPORT
# =============================================================================

def quality_report(data: dict) -> pd.DataFrame:
    """Print and return a summary quality table for all companies."""
    print("\n" + "="*65)
    print("  DATA QUALITY REPORT")
    print("="*65)

    rows = []
    for ticker, df in data.items():
        bday_idx      = pd.bdate_range(df.index.min(), df.index.max())
        trading_days  = len(df)
        calendar_days = (df.index.max() - df.index.min()).days
        gaps          = len(bday_idx) - trading_days
        missing       = df.isnull().sum().sum()
        dups          = df.index.duplicated().sum()
        price_viol    = ((df["close"] < df["low"]) |
                         (df["close"] > df["high"])).sum()
        zero_vol      = (df["volume"] == 0).sum()

        rows.append({
            "Ticker"        : ticker,
            "Trading Days"  : trading_days,
            "Date From"     : df.index.min().date(),
            "Date To"       : df.index.max().date(),
            "Calendar Days" : calendar_days,
            "BD Gaps"       : gaps,
            "Missing Values": missing,
            "Duplicates"    : dups,
            "Price Violations": price_viol,
            "Zero Volume"   : zero_vol,
        })
        print(f"\n  {ticker}")
        print(f"    Trading days   : {trading_days}")
        print(f"    Date range     : {df.index.min().date()} → {df.index.max().date()}")
        print(f"    BD gaps        : {gaps}")
        print(f"    Missing values : {missing}")
        print(f"    Price violations: {price_viol}")

    summary = pd.DataFrame(rows).set_index("Ticker")
    print("\n" + "="*65)
    return summary


# =============================================================================
# 3.  CLEANING
# =============================================================================

def clean_company(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Full cleaning pipeline for one company."""

    # 3a – duplicates
    df = df[~df.index.duplicated(keep="last")]

    # 3b – reindex to business days
    bday_idx = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(bday_idx)

    # 3c – negative prices → NaN
    price_cols = ["open", "high", "low", "close", "adj_close"]
    for col in price_cols:
        df.loc[df[col] < 0, col] = np.nan
    df["volume"] = df["volume"].clip(lower=0)

    # Fix OHLC consistency
    swap = df["high"] < df["low"]
    if swap.sum():
        df.loc[swap, ["high", "low"]] = df.loc[swap, ["low", "high"]].values

    # 3d – forward-fill short gaps, interpolate medium gaps
    df[price_cols] = (df[price_cols]
                      .ffill(limit=2)
                      .interpolate(method="linear", limit=5,
                                   limit_direction="forward"))
    df["volume"] = df["volume"].ffill(limit=2).fillna(0)

    # 3e – drop remaining missing close
    before = len(df)
    df = df.dropna(subset=["close"])
    print(f"  [{ticker}] Clean: {len(df)} rows (dropped {before - len(df)})")
    return df


def clean_all(data: dict) -> dict:
    print("\n[CLEAN] Cleaning all companies …")
    return {ticker: clean_company(df, ticker) for ticker, df in data.items()}


# =============================================================================
# 4.  FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Engineer all ML features for one company."""
    f = df.copy()

    # ── 4a  Returns ──────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 20]:
        f[f"return_{lag}d"]     = f["adj_close"].pct_change(lag)
    f["log_return_1d"]          = np.log(f["adj_close"] /
                                         f["adj_close"].shift(1))
    f["overnight_gap"]          = (f["open"] - f["close"].shift(1)) / \
                                   f["close"].shift(1)
    f["intraday_range"]         = (f["high"] - f["low"]) / f["close"]
    f["close_open_return"]      = (f["close"] - f["open"]) / f["open"]

    # ── 4b  Rolling statistics ────────────────────────────────────────────
    for w in ROLLING_WINDOWS:
        f[f"ma_{w}"]            = f["close"].rolling(w).mean()
        f[f"std_{w}"]           = f["close"].rolling(w).std()
        f[f"min_{w}"]           = f["close"].rolling(w).min()
        f[f"max_{w}"]           = f["close"].rolling(w).max()
        f[f"price_rel_ma_{w}"]  = f["close"] / f[f"ma_{w}"] - 1
        f[f"vol_ma_{w}"]        = f["volume"].rolling(w).mean()
        f[f"vol_std_{w}"]       = f["volume"].rolling(w).std()
        f[f"return_skew_{w}"]   = f["log_return_1d"].rolling(w).skew()
        f[f"return_kurt_{w}"]   = f["log_return_1d"].rolling(w).kurt()

    # ── 4c  Technical indicators ──────────────────────────────────────────
    if HAS_TA:
        f["ema_12"]             = ta.trend.ema_indicator(f["close"], 12)
        f["ema_26"]             = ta.trend.ema_indicator(f["close"], 26)
        f["ema_50"]             = ta.trend.ema_indicator(f["close"], 50)
        f["macd"]               = ta.trend.macd(f["close"])
        f["macd_signal"]        = ta.trend.macd_signal(f["close"])
        f["macd_diff"]          = ta.trend.macd_diff(f["close"])
        f["adx"]                = ta.trend.adx(f["high"], f["low"], f["close"])
        f["rsi_7"]              = ta.momentum.rsi(f["close"], 7)
        f["rsi_14"]             = ta.momentum.rsi(f["close"], 14)
        f["rsi_21"]             = ta.momentum.rsi(f["close"], 21)
        stoch = ta.momentum.StochasticOscillator(f["high"], f["low"], f["close"])
        f["stoch_k"]            = stoch.stoch()
        f["stoch_d"]            = stoch.stoch_signal()
        f["cci"]                = ta.trend.cci(f["high"], f["low"], f["close"])
        f["williams_r"]         = ta.momentum.williams_r(f["high"], f["low"], f["close"])
        bb = ta.volatility.BollingerBands(f["close"])
        f["bb_upper"]           = bb.bollinger_hband()
        f["bb_lower"]           = bb.bollinger_lband()
        f["bb_pct"]             = bb.bollinger_pband()
        f["bb_width"]           = bb.bollinger_wband()
        f["atr_14"]             = ta.volatility.average_true_range(
                                    f["high"], f["low"], f["close"], 14)
        f["obv"]                = ta.volume.on_balance_volume(f["close"], f["volume"])
        f["cmf"]                = ta.volume.chaikin_money_flow(
                                    f["high"], f["low"], f["close"], f["volume"])
        f["vwap_proxy"]         = (f["volume"] * (f["high"] + f["low"] + f["close"]) / 3
                                   ).cumsum() / f["volume"].cumsum()
    else:
        # Manual fallback
        ema12                   = f["close"].ewm(span=12).mean()
        ema26                   = f["close"].ewm(span=26).mean()
        f["macd"]               = ema12 - ema26
        f["macd_signal"]        = f["macd"].ewm(span=9).mean()
        f["macd_diff"]          = f["macd"] - f["macd_signal"]
        delta                   = f["close"].diff()
        gain                    = delta.clip(lower=0).rolling(14).mean()
        loss                    = (-delta.clip(upper=0)).rolling(14).mean()
        f["rsi_14"]             = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        ma20                    = f["close"].rolling(20).mean()
        std20                   = f["close"].rolling(20).std()
        f["bb_upper"]           = ma20 + 2 * std20
        f["bb_lower"]           = ma20 - 2 * std20
        f["bb_pct"]             = (f["close"] - f["bb_lower"]) / \
                                   (f["bb_upper"] - f["bb_lower"])
        f["bb_width"]           = (f["bb_upper"] - f["bb_lower"]) / ma20

    # ── 4d  Volume features ───────────────────────────────────────────────
    f["log_volume"]             = np.log1p(f["volume"])
    f["vol_change_1d"]          = f["volume"].pct_change(1)
    f["vol_surge_20"]           = f["volume"] / f["volume"].rolling(20).mean()
    f["vol_price_ratio"]        = f["volume"] / f["close"]

    # ── 4e  Calendar / seasonality ────────────────────────────────────────
    f["day_of_week"]            = f.index.dayofweek
    f["month"]                  = f.index.month
    f["quarter"]                = f.index.quarter
    f["is_month_end"]           = f.index.is_month_end.astype(int)
    f["is_month_start"]         = f.index.is_month_start.astype(int)
    f["is_quarter_end"]         = f.index.is_quarter_end.astype(int)
    # Fiscal year (Nepal: mid-July → mid-July)
    f["nepali_fiscal_year"]     = f.index.year + (f.index.month >= 7).astype(int)

    # ── 4f  Price structure ratios ────────────────────────────────────────
    f["close_open_ratio"]       = f["close"] / f["open"]
    f["high_close_ratio"]       = f["high"]  / f["close"]
    f["low_close_ratio"]        = f["low"]   / f["close"]
    f["adj_factor"]             = f["adj_close"] / f["close"]

    # ── 4g  Lag features (for LSTM context) ──────────────────────────────
    for lag in [1, 2, 3, 5]:
        f[f"close_lag_{lag}"]   = f["close"].shift(lag)
        f[f"vol_lag_{lag}"]     = f["volume"].shift(lag)

    return f


def engineer_all(data: dict) -> dict:
    print("\n[FEATURES] Engineering features …")
    result = {}
    for ticker, df in data.items():
        result[ticker] = engineer_features(df, ticker)
        print(f"  [{ticker}] {result[ticker].shape[1]} features")
    return result


# =============================================================================
# 5.  TARGET VARIABLES
# =============================================================================

def create_targets(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> pd.DataFrame:
    """Add regression and classification targets."""
    future = df["close"].shift(-horizon)
    df[f"target_return_{horizon}d"]    = (future - df["close"]) / df["close"]
    df[f"target_log_return_{horizon}d"]= np.log(future / df["close"])
    df[f"target_direction_{horizon}d"] = (future > df["close"]).astype(int)
    return df


# =============================================================================
# 6.  OUTLIER WINSORISATION
# =============================================================================

def winsorise(df: pd.DataFrame, cols: list, iqr_factor=3.0) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr    = q3 - q1
        df[col]= df[col].clip(q1 - iqr_factor * iqr,
                               q3 + iqr_factor * iqr)
    return df


# =============================================================================
# 7.  TEMPORAL SPLIT
# =============================================================================

def temporal_split(df: pd.DataFrame,
                   train_r=TRAIN_RATIO, val_r=VAL_RATIO):
    n       = len(df)
    i_train = int(n * train_r)
    i_val   = int(n * (train_r + val_r))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


# =============================================================================
# 8.  FEATURE SELECTION & SCALING
# =============================================================================

def get_feature_cols(df: pd.DataFrame) -> list:
    skip = {"open", "high", "low", "close", "adj_close", "volume"}
    skip |= {c for c in df.columns if c.startswith("target_")}
    return [c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def scale_split(train, val, test, feature_cols, target_col):
    """RobustScaler fit on train only. Returns arrays + scaler."""
    def clean(d):
        return d[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler  = RobustScaler()
    X_train = scaler.fit_transform(clean(train))
    X_val   = scaler.transform(clean(val))
    X_test  = scaler.transform(clean(test))
    y_train = train[target_col].values
    y_val   = val[target_col].values
    y_test  = test[target_col].values
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# =============================================================================
# 9.  TENSORFLOW SEQUENCE BUILDER
# =============================================================================

def make_sequences(X: np.ndarray, y: np.ndarray,
                   seq_len: int = SEQ_LENGTH):
    """
    Convert flat feature matrix → overlapping sequences for LSTM/GRU.
    Returns:
        Xs : shape (samples, seq_len, n_features)
        ys : shape (samples,)
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def build_lstm_model(seq_len: int, n_features: int,
                     units=(128, 64), dropout=0.3) -> tf.keras.Model:
    """
    Stacked LSTM model with Batch Normalisation and Dropout.
    Suitable for regression (target_return) or classification (target_direction).
    """
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(units[0], return_sequences=True),
        BatchNormalization(),
        Dropout(dropout),
        LSTM(units[1], return_sequences=False),
        BatchNormalization(),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dropout(dropout / 2),
        Dense(1),                   # change to Dense(1, activation='sigmoid') for classification
    ], name="RADHI_LSTM")
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def build_gru_model(seq_len: int, n_features: int,
                    units=(128, 64), dropout=0.3) -> tf.keras.Model:
    """GRU variant — often faster to train than LSTM on smaller datasets."""
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        GRU(units[0], return_sequences=True),
        BatchNormalization(),
        Dropout(dropout),
        GRU(units[1], return_sequences=False),
        BatchNormalization(),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dropout(dropout / 2),
        Dense(1),
    ], name="RADHI_GRU")
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def get_tf_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
    ]


# =============================================================================
# 10.  EXTERNAL DATA MERGE HOOK
# =============================================================================

def merge_external(df: pd.DataFrame, ticker: str,
                   ext: dict = None) -> pd.DataFrame:
    """
    Merge fundamentals, rainfall, and production DataFrames when available.
    Each external DataFrame must have a DateTimeIndex.
    Annual/quarterly data is forward-filled to daily frequency.

    Example usage (once you have the data):
    ──────────────────────────────────────────────────────────────────────
    fundamentals = pd.read_csv("RADHI_fundamentals.csv",
                               index_col="date", parse_dates=True)
    rainfall     = pd.read_csv("rainfall_monthly.csv",
                               index_col="date", parse_dates=True)
    production   = pd.read_csv("production_annual.csv",
                               index_col="date", parse_dates=True)

    EXTERNAL_DATA["RADHI"]["fundamentals"] = fundamentals
    EXTERNAL_DATA["RADHI"]["rainfall"]     = rainfall
    EXTERNAL_DATA["RADHI"]["production"]   = production
    ──────────────────────────────────────────────────────────────────────
    Expected column names per source:
      fundamentals : eps, pe_ratio, book_value, roe, dps, debt_equity
      rainfall     : rainfall_mm
      production   : production_gwh (annual electricity generation)
    """
    if ext is None:
        return df
    for name, ext_df in ext.items():
        if ext_df is None:
            continue
        daily = ext_df.resample("B").ffill()
        df    = df.join(daily, how="left")
        print(f"  [{ticker}] Merged {name}: {daily.shape[1]} columns")
    return df


# =============================================================================
# 11.  THESIS-QUALITY PLOTS
# =============================================================================

# ── Helper ────────────────────────────────────────────────────────────────────
def _save(fig, name: str):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [PLOT] Saved → {path.name}")


# ── Plot 1: Individual company overview ───────────────────────────────────────
def plot_company_overview(clean_data: dict, feat_data: dict):
    """
    Per-company: Price + Volume + RSI + Bollinger Bands + MACD
    One figure per company (thesis Chapter 3 — Data Description).
    """
    for ticker, df in feat_data.items():
        color = COMPANY_COLORS.get(ticker, "#2C3E50")
        raw   = clean_data[ticker]

        fig = plt.figure(figsize=(16, 14))
        gs  = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1.2, 1.2],
                                hspace=0.08)
        axes = [fig.add_subplot(gs[i]) for i in range(4)]

        fig.suptitle(f"{ticker} — Price & Technical Overview\n"
                     f"({raw.index.min().strftime('%b %Y')} – "
                     f"{raw.index.max().strftime('%b %Y')})",
                     fontsize=15, fontweight="bold", y=0.98)

        # Panel 1: Price + Bollinger Bands + EMAs
        ax = axes[0]
        ax.plot(df.index, df["close"], color=color, lw=1.2, label="Close", zorder=3)
        if "bb_upper" in df:
            ax.fill_between(df.index, df["bb_lower"], df["bb_upper"],
                            alpha=0.15, color=color, label="BB (2σ)")
            ax.plot(df.index, df["bb_upper"], lw=0.6, ls="--",
                    color=color, alpha=0.6)
            ax.plot(df.index, df["bb_lower"], lw=0.6, ls="--",
                    color=color, alpha=0.6)
        if "ma_20" in df:
            ax.plot(df.index, df["ma_20"], lw=1, ls="-.",
                    color=ACCENT_COLOR, label="MA-20", alpha=0.9)
        if "ma_50" in df:
            ax.plot(df.index, df["ma_50"], lw=1, ls=":",
                    color="#8E44AD", label="MA-50", alpha=0.9)
        ax.set_ylabel("Price (NPR)")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xticklabels([])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

        # Panel 2: Volume
        ax = axes[1]
        vol_ma = df["volume"].rolling(20).mean()
        ax.bar(df.index, df["volume"], width=1.2,
               color=color, alpha=0.35, label="Volume")
        ax.plot(df.index, vol_ma, color=ACCENT_COLOR, lw=1,
                label="MA-20 Vol")
        ax.set_ylabel("Volume")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xticklabels([])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

        # Panel 3: RSI
        ax = axes[2]
        if "rsi_14" in df:
            rsi = df["rsi_14"].dropna()
            ax.plot(rsi.index, rsi, color="#2980B9", lw=0.9, label="RSI-14")
            ax.axhline(70, ls="--", color="#E74C3C", lw=1, alpha=0.8,
                       label="Overbought (70)")
            ax.axhline(30, ls="--", color="#27AE60", lw=1, alpha=0.8,
                       label="Oversold (30)")
            ax.axhline(50, ls=":", color="grey", lw=0.7, alpha=0.6)
            ax.fill_between(rsi.index, 70, rsi.clip(lower=70),
                            alpha=0.2, color="#E74C3C")
            ax.fill_between(rsi.index, rsi.clip(upper=30), 30,
                            alpha=0.2, color="#27AE60")
        ax.set_ylim(0, 100)
        ax.set_ylabel("RSI")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xticklabels([])

        # Panel 4: MACD
        ax = axes[3]
        if "macd" in df:
            macd_v = df["macd"].fillna(0)
            sig_v  = df["macd_signal"].fillna(0)
            diff_v = df["macd_diff"].fillna(0)
            ax.plot(df.index, macd_v, color="#2980B9", lw=0.9, label="MACD")
            ax.plot(df.index, sig_v,  color=ACCENT_COLOR, lw=0.9,
                    ls="--", label="Signal")
            bar_colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in diff_v]
            ax.bar(df.index, diff_v, width=1.2, color=bar_colors,
                   alpha=0.5, label="Histogram")
            ax.axhline(0, color="grey", lw=0.7)
        ax.set_ylabel("MACD")
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        _save(fig, f"{ticker}_01_overview.png")


# ── Plot 2: Return Distribution Analysis ─────────────────────────────────────
def plot_return_distributions(feat_data: dict):
    """
    Daily return distributions per company — histogram + KDE + Q-Q plot.
    Thesis Chapter 3 — Statistical Properties of Returns.
    """
    n = len(feat_data)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Daily Log-Return Distributions\n(Hydropower Companies)",
                 fontsize=14, fontweight="bold")

    for i, (ticker, df) in enumerate(feat_data.items()):
        color  = COMPANY_COLORS.get(ticker, "#2C3E50")
        ret    = df["log_return_1d"].dropna()
        mu, sd = ret.mean(), ret.std()
        skew   = ret.skew()
        kurt   = ret.kurtosis()

        # Histogram + KDE
        ax = axes[i, 0]
        ax.hist(ret, bins=80, density=True,
                color=color, alpha=0.6, edgecolor="none",
                label=f"Obs  μ={mu:.4f}  σ={sd:.4f}")
        xr = np.linspace(ret.min(), ret.max(), 300)
        ax.plot(xr, stats.norm.pdf(xr, mu, sd),
                color="#E74C3C", lw=1.5, ls="--", label="Normal fit")
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(ret)
        ax.plot(xr, kde(xr), color=ACCENT_COLOR, lw=1.5, label="KDE")
        ax.set_title(f"{ticker} — Return Distribution")
        ax.set_xlabel("Log Return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.text(0.98, 0.95,
                f"Skew: {skew:.3f}\nKurt: {kurt:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                      fc="white", alpha=0.8))

        # Q-Q plot
        ax = axes[i, 1]
        (osm, osr), (slope, intercept, _) = stats.probplot(ret, dist="norm")
        ax.scatter(osm, osr, s=4, color=color, alpha=0.4)
        ax.plot(osm, slope * np.array(osm) + intercept,
                color="#E74C3C", lw=1.5, ls="--", label="Normal ref.")
        ax.set_title(f"{ticker} — Q-Q Plot")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.legend(fontsize=9)

    _save(fig, "ALL_02_return_distributions.png")


# ── Plot 3: Cross-company correlation ────────────────────────────────────────
def plot_cross_correlation(feat_data: dict):
    """
    Cross-company return correlations — heatmap + rolling 60-day correlation.
    Thesis Chapter 4 — Inter-company Analysis.
    """
    tickers = list(feat_data.keys())
    if len(tickers) < 2:
        print("  [PLOT] Skipping cross-correlation (only one company loaded)")
        return

    returns = pd.DataFrame({t: feat_data[t]["log_return_1d"]
                             for t in tickers}).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-Company Return Correlation\n(Hydropower Sector)",
                 fontsize=14, fontweight="bold")

    # Static heatmap
    corr = returns.corr()
    mask = np.triu(np.ones_like(corr), k=1)
    sns.heatmap(corr, ax=axes[0], annot=True, fmt=".3f",
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                linewidths=0.5, square=True,
                annot_kws={"size": 12, "weight": "bold"})
    axes[0].set_title("Pearson Correlation Matrix")

    # Rolling 60-day pairwise correlations
    ax = axes[1]
    pairs = [(tickers[i], tickers[j])
             for i in range(len(tickers))
             for j in range(i + 1, len(tickers))]
    colors_p = ["#1B4F72", "#1E8449", "#922B21",
                "#8E44AD", "#D35400", "#17A589"]
    for idx, (t1, t2) in enumerate(pairs):
        roll_corr = returns[t1].rolling(60).corr(returns[t2])
        ax.plot(roll_corr.index, roll_corr,
                label=f"{t1}–{t2}", color=colors_p[idx], lw=1.2)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_title("60-Day Rolling Pairwise Correlation")
    ax.set_ylabel("Pearson r")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    _save(fig, "ALL_03_cross_correlation.png")


# ── Plot 4: Feature importance proxy (correlation with target) ────────────────
def plot_feature_target_correlation(feat_data: dict, target_col: str):
    """
    Bar chart of top-20 features most correlated with the target.
    One chart per company.
    Thesis Chapter 4 — Feature Analysis.
    """
    for ticker, df in feat_data.items():
        color    = COMPANY_COLORS.get(ticker, "#2C3E50")
        feat_cols = get_feature_cols(df)
        sub       = df[feat_cols + [target_col]].dropna()
        corr      = sub.corr()[target_col].drop(target_col)
        top20     = corr.abs().nlargest(20)
        vals      = corr[top20.index]

        fig, ax = plt.subplots(figsize=(10, 7))
        bar_colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in vals]
        bars = ax.barh(vals.index, vals.values,
                       color=bar_colors, edgecolor="none", height=0.7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{ticker} — Top-20 Feature Correlations with\n"
                     f"`{target_col}`",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Pearson Correlation Coefficient")
        ax.invert_yaxis()

        # Annotate values
        for bar, val in zip(bars, vals.values):
            xpos = val + 0.003 if val >= 0 else val - 0.003
            ha   = "left" if val >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha=ha, fontsize=8)

        ax.text(0.98, 0.02,
                f"Target: {PREDICTION_HORIZON}-day return",
                transform=ax.transAxes, ha="right", fontsize=9,
                style="italic", color="grey")
        plt.tight_layout()
        _save(fig, f"{ticker}_04_feature_correlation.png")


# ── Plot 5: Train / Val / Test split visualisation ────────────────────────────
def plot_data_splits(feat_data: dict, target_col: str):
    """
    Visualise how the dataset is split across time.
    Thesis Chapter 3 — Methodology.
    """
    n   = len(feat_data)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("Train / Validation / Test Split\n(Temporal — No Shuffling)",
                 fontsize=14, fontweight="bold")

    split_colors = {"Train": "#1B4F72", "Validation": "#1E8449",
                    "Test" : "#922B21"}

    for ax, (ticker, df) in zip(axes, feat_data.items()):
        df_clean = df.dropna(subset=[target_col])
        train, val, test = temporal_split(df_clean)

        for split_name, split_df, alpha in [
                ("Train", train, 0.7),
                ("Validation", val, 0.8),
                ("Test", test, 0.9)]:
            c = split_colors[split_name]
            ax.fill_between(split_df.index, split_df["close"],
                            alpha=0.25, color=c)
            ax.plot(split_df.index, split_df["close"],
                    color=c, lw=0.9,
                    label=f"{split_name} ({len(split_df)} days, "
                          f"{split_df.index[0].strftime('%b %y')}–"
                          f"{split_df.index[-1].strftime('%b %y')})")

        ax.set_title(f"{ticker}")
        ax.set_ylabel("Close Price (NPR)")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    plt.tight_layout()
    _save(fig, "ALL_05_data_splits.png")


# ── Plot 6: Volatility regime ─────────────────────────────────────────────────
def plot_volatility(feat_data: dict):
    """
    Rolling 20-day realised volatility (annualised) for all companies.
    Thesis Chapter 3 — Descriptive Statistics.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Annualised 20-Day Realised Volatility\n(Hydropower Companies)",
                 fontsize=14, fontweight="bold")

    for ticker, df in feat_data.items():
        color = COMPANY_COLORS.get(ticker, "#2C3E50")
        rv    = df["log_return_1d"].rolling(20).std() * np.sqrt(252) * 100
        ax.plot(rv.index, rv, color=color, lw=1, alpha=0.85, label=ticker)

    ax.set_ylabel("Volatility (%)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.tight_layout()
    _save(fig, "ALL_06_volatility.png")


# ── Plot 7: Seasonal / monthly patterns ──────────────────────────────────────
def plot_seasonal_patterns(feat_data: dict):
    """
    Average monthly returns heatmap (year × month).
    Thesis Chapter 4 — Seasonal Effects (Nepal's monsoon pattern).
    """
    n   = len(feat_data)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle("Average Monthly Log-Returns (Year × Month)\n"
                 "— Captures Monsoon Seasonality",
                 fontsize=14, fontweight="bold")

    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    for ax, (ticker, df) in zip(axes, feat_data.items()):
        pivot = (df["log_return_1d"]
                 .groupby([df.index.year, df.index.month])
                 .mean()
                 .unstack(level=1) * 100)
        pivot.columns = months[:len(pivot.columns)]

        sns.heatmap(pivot, ax=ax, cmap="RdYlGn", center=0,
                    fmt=".1f", annot=True, annot_kws={"size": 7},
                    linewidths=0.3, cbar_kws={"label": "Return (%)"})
        ax.set_title(f"{ticker}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")

    plt.tight_layout()
    _save(fig, "ALL_07_seasonal_heatmap.png")


# ── Plot 8: Preprocessing summary statistics table ───────────────────────────
def plot_summary_stats(clean_data: dict, feat_data: dict):
    """
    Descriptive statistics table saved as a figure.
    Thesis Chapter 3 — Data Summary Table.
    """
    rows = []
    for ticker in clean_data:
        df  = clean_data[ticker]
        ret = feat_data[ticker]["log_return_1d"].dropna()
        rows.append({
            "Company"       : ticker,
            "Obs."          : f"{len(df):,}",
            "From"          : df.index.min().strftime("%Y-%m-%d"),
            "To"            : df.index.max().strftime("%Y-%m-%d"),
            "Mean Close"    : f"{df['close'].mean():,.1f}",
            "Std Close"     : f"{df['close'].std():,.1f}",
            "Min Close"     : f"{df['close'].min():,.1f}",
            "Max Close"     : f"{df['close'].max():,.1f}",
            "Mean Return"   : f"{ret.mean():.5f}",
            "Std Return"    : f"{ret.std():.5f}",
            "Skewness"      : f"{ret.skew():.3f}",
            "Kurtosis"      : f"{ret.kurtosis():.3f}",
        })

    df_table = pd.DataFrame(rows).set_index("Company")
    fig, ax  = plt.subplots(figsize=(18, 2 + len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText  = df_table.values,
        colLabels = df_table.columns,
        rowLabels = df_table.index,
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    # Style header
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#1B4F72")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EBF5FB")

    fig.suptitle("Descriptive Statistics — NEPSE Hydropower Companies",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    _save(fig, "ALL_08_summary_statistics.png")


# ── Plot 9: Correlation feature heatmap ──────────────────────────────────────
def plot_feature_heatmap(feat_data: dict, n_features: int = 25):
    """
    Heatmap of top-N feature × feature correlations.
    Thesis Chapter 4 — Multicollinearity.
    """
    for ticker, df in feat_data.items():
        feat_cols = get_feature_cols(df)[:n_features]
        sub       = df[feat_cols].dropna()
        corr      = sub.corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr, ax=ax, cmap="RdYlGn", center=0,
                    vmin=-1, vmax=1, annot=False,
                    linewidths=0.3, square=True,
                    xticklabels=True, yticklabels=True)
        ax.set_title(f"{ticker} — Feature Correlation Matrix (Top {n_features})",
                     fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        _save(fig, f"{ticker}_09_feature_heatmap.png")


# ── Plot 10: Preprocessing pipeline diagram ──────────────────────────────────
def plot_pipeline_diagram():
    """
    Visual flowchart of the preprocessing pipeline.
    Thesis Chapter 3 — Methodology.
    """
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    steps = [
        ("Raw OHLCV Data\n(6 CSV files per company)", "#1B4F72", "white"),
        ("Step 1: Load & Merge\nDate parsing · comma-formatted volume", "#2471A3", "white"),
        ("Step 2: Quality Assessment\nMissing · Duplicates · OHLC violations", "#2980B9", "white"),
        ("Step 3: Cleaning\nBusiness-day reindex · Forward-fill · Interpolate", "#5DADE2", "white"),
        ("Step 4: Feature Engineering\n70+ features: Returns · TA · Volume · Calendar", "#1E8449", "white"),
        ("Step 5: External Data Merge\nFundamentals · Rainfall · Production (hook)", "#27AE60", "white"),
        ("Step 6: Target Creation\n5-day return · log return · direction", "#F39C12", "white"),
        ("Step 7: Outlier Winsorisation\nIQR × 3.0 — preserves genuine price spikes", "#D35400", "white"),
        ("Step 8: Temporal Split\n70% Train · 15% Val · 15% Test (no shuffle)", "#922B21", "white"),
        ("Step 9: Robust Scaling\nFit on Train only · Transform Val & Test", "#7D3C98", "white"),
        ("Step 10: LSTM Sequences\nShape: (samples, 60, n_features)", "#6C3483", "white"),
        ("TensorFlow-Ready Arrays\nX_train, X_val, X_test, y_*", "#1A5276", "white"),
    ]

    y_start = 0.97
    dy      = 0.075
    for i, (text, fc, tc) in enumerate(steps):
        y = y_start - i * dy
        arrow = "▼" if i < len(steps) - 1 else ""
        bbox  = dict(boxstyle="round,pad=0.5", fc=fc, ec="none", alpha=0.92)
        ax.text(0.5, y, text, transform=ax.transAxes,
                ha="center", va="top", fontsize=10,
                color=tc, bbox=bbox, fontweight="bold" if i in [0, 11] else "normal")
        if arrow:
            ax.text(0.5, y - dy * 0.45, arrow,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="#555555")

    ax.set_title("ML Preprocessing Pipeline\nNEPSE Hydropower Stock Analysis",
                 fontsize=14, fontweight="bold", pad=15)
    _save(fig, "ALL_00_pipeline_diagram.png")


# =============================================================================
# 12.  MAIN PIPELINE
# =============================================================================

def run_pipeline():
    print("\n" + "="*65)
    print("  NEPSE HYDROPOWER — MULTI-COMPANY ML PREPROCESSING PIPELINE")
    print("="*65)

    target_col = f"target_return_{PREDICTION_HORIZON}d"

    # ── Load ────────────────────────────────────────────────────────────────
    raw_data   = load_all_companies(COMPANIES)

    # ── Quality report ──────────────────────────────────────────────────────
    quality_df = quality_report(raw_data)
    quality_df.to_csv(OUTPUT_DIR / "quality_report.csv")

    # ── Clean ───────────────────────────────────────────────────────────────
    clean_data = clean_all(raw_data)

    # ── Feature engineering ─────────────────────────────────────────────────
    feat_data  = engineer_all(clean_data)

    # ── External data merge (no-op until DataFrames provided) ───────────────
    print("\n[EXT] Merging external data …")
    for ticker in feat_data:
        ext = EXTERNAL_DATA.get(ticker)
        feat_data[ticker] = merge_external(feat_data[ticker], ticker, ext)

    # ── Targets ─────────────────────────────────────────────────────────────
    print("\n[TARGET] Creating targets …")
    for ticker in feat_data:
        feat_data[ticker] = create_targets(feat_data[ticker], PREDICTION_HORIZON)
        feat_data[ticker] = feat_data[ticker].dropna(subset=[target_col])
        print(f"  [{ticker}] {len(feat_data[ticker])} rows after target creation")

    # ── Outlier winsorisation ────────────────────────────────────────────────
    print("\n[OUTLIERS] Winsorising …")
    for ticker in feat_data:
        ret_cols = [c for c in feat_data[ticker].columns
                    if "return" in c and "target" not in c]
        feat_data[ticker] = winsorise(feat_data[ticker], ret_cols)

    # ── Per-company split, scale, TF sequences ──────────────────────────────
    print("\n[SPLIT + SCALE + TF SEQUENCES]")
    tf_ready   = {}
    scalers    = {}

    for ticker, df in feat_data.items():
        feat_cols = get_feature_cols(df)
        train, val, test = temporal_split(df)

        print(f"\n  [{ticker}]")
        print(f"    Train : {len(train)} ({train.index[0].date()} → {train.index[-1].date()})")
        print(f"    Val   : {len(val)}   ({val.index[0].date()} → {val.index[-1].date()})")
        print(f"    Test  : {len(test)}  ({test.index[0].date()} → {test.index[-1].date()})")

        (X_train, X_val, X_test,
         y_train, y_val, y_test, scaler) = scale_split(
            train, val, test, feat_cols, target_col)
        scalers[ticker] = scaler

        # TensorFlow sequences
        Xs_train, ys_train = make_sequences(X_train, y_train, SEQ_LENGTH)
        Xs_val,   ys_val   = make_sequences(X_val,   y_val,   SEQ_LENGTH)
        Xs_test,  ys_test  = make_sequences(X_test,  y_test,  SEQ_LENGTH)

        print(f"    TF train seq : {Xs_train.shape}")
        print(f"    TF val   seq : {Xs_val.shape}")
        print(f"    TF test  seq : {Xs_test.shape}")

        # Build models (not trained here — training is the next step)
        n_feat  = Xs_train.shape[2]
        lstm    = build_lstm_model(SEQ_LENGTH, n_feat)
        gru     = build_gru_model(SEQ_LENGTH, n_feat)

        tf_ready[ticker] = {
            "Xs_train": Xs_train, "ys_train": ys_train,
            "Xs_val"  : Xs_val,   "ys_val"  : ys_val,
            "Xs_test" : Xs_test,  "ys_test" : ys_test,
            "lstm_model": lstm,   "gru_model": gru,
            "feature_cols": feat_cols,
            "scaler": scaler,
            "tscv": TimeSeriesSplit(n_splits=N_CV_FOLDS,
                                    gap=PREDICTION_HORIZON),
        }

        # Save arrays
        np.savez(OUTPUT_DIR / f"{ticker}_arrays.npz",
                 X_train=X_train, X_val=X_val,   X_test=X_test,
                 y_train=y_train, y_val=y_val,    y_test=y_test,
                 Xs_train=Xs_train, Xs_val=Xs_val, Xs_test=Xs_test,
                 ys_train=ys_train, ys_val=ys_val, ys_test=ys_test)

        # Save feature-engineered CSV
        feat_data[ticker].to_csv(OUTPUT_DIR / f"{ticker}_features.csv")
        pd.Series(feat_cols, name="feature").to_csv(
            OUTPUT_DIR / f"{ticker}_feature_list.csv", index=False)

    # ── Thesis plots ────────────────────────────────────────────────────────
    print("\n[PLOTS] Generating thesis-quality figures …")
    plot_pipeline_diagram()
    plot_company_overview(clean_data, feat_data)
    plot_return_distributions(feat_data)
    plot_cross_correlation(feat_data)
    plot_feature_target_correlation(feat_data, target_col)
    plot_data_splits(feat_data, target_col)
    plot_volatility(feat_data)
    plot_seasonal_patterns(feat_data)
    plot_summary_stats(clean_data, feat_data)
    plot_feature_heatmap(feat_data)

    # ── Model summaries ─────────────────────────────────────────────────────
    print("\n[MODELS] Model architectures (first company):")
    first = list(tf_ready.keys())[0]
    tf_ready[first]["lstm_model"].summary()
    tf_ready[first]["gru_model"].summary()

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PIPELINE COMPLETE")
    print("="*65)
    print(f"\n  Output directory : {OUTPUT_DIR}")
    print(f"\n  Files generated per company:")
    print(f"    {{ticker}}_arrays.npz      — NumPy arrays (flat + sequences)")
    print(f"    {{ticker}}_features.csv    — Full feature-engineered dataset")
    print(f"    {{ticker}}_feature_list.csv")
    print(f"\n  Thesis figures (PNG, 180 DPI):")
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"    {p.name}")
    print(f"\n  Next steps:")
    print(f"    1. Train LSTM/GRU via tf_ready[ticker]['lstm_model'].fit(...)")
    print(f"    2. Add fundamentals/rainfall/production to EXTERNAL_DATA dict")
    print(f"    3. Run XGBoost on flat arrays (X_train / X_val / X_test)")
    print(f"    4. SHAP explainability on trained models")

    return feat_data, tf_ready


# =============================================================================
if __name__ == "__main__":
    feat_data, tf_ready = run_pipeline()