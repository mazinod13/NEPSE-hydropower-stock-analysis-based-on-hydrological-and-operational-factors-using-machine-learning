"""
Hydrology, Production, and Market Movement Study
================================================

Builds hydrology-aware daily market features and compares:
    1. Regularized LSTM on 60-day sequences
    2. Gradient Boosting on flat daily features

The script merges annual rainfall and production features into each company's
daily market feature table by Company + Year, then trains models to predict
the next 5 trading-day return. Directional accuracy is reported from the sign
of the predicted return.

Inputs:
    Output/{COMPANY}_features.csv
    Features/Production/production_features.csv
    Features/Rainfall/rainfall_features.csv

Outputs:
    Features/HydroMarket/{COMPANY}_hydro_market_features.csv
    Output/HydroMarketStudy/{COMPANY}_hydro_market_arrays.npz
    Output/HydroMarketStudy/hydro_market_model_results.csv
    Output/HydroMarketStudy/hydro_market_relation_summary.csv
    Output/HydroMarketStudy/*.png

Run:
    python hybrid_hydro_market_model.py

Quick smoke test without training:
    python hybrid_hydro_market_model.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
import joblib


COMPANIES = ["BARUN", "CHCL", "RADHI"]
DEFAULT_TARGET_HORIZON = 5
TARGET_HORIZON = DEFAULT_TARGET_HORIZON
TARGET_COL = f"target_return_{TARGET_HORIZON}d"
TARGET_LOG_COL = f"target_log_return_{TARGET_HORIZON}d"
TARGET_DIRECTION_COL = f"target_direction_{TARGET_HORIZON}d"
SEQ_LENGTH = 60
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42

PRICE_FEATURE_DIR = Path("Output")
PRODUCTION_FEATURE_PATH = Path("Features") / "Production" / "production_features.csv"
RAINFALL_FEATURE_PATH = Path("Features") / "Rainfall" / "rainfall_features.csv"
FEATURE_OUTPUT_DIR = Path("Features") / "HydroMarket"
STUDY_OUTPUT_DIR = Path("Output") / "HydroMarketStudy"

HYDRO_PRODUCTION_COLUMNS = [
    "Energy Generation (GWh)",
    "generation_per_trading_day",
    "generation_gwh_lag_1y",
    "generation_gwh_lag_2y",
    "generation_gwh_yoy_change",
    "generation_gwh_yoy_pct_change",
    "generation_gwh_rolling_3y_mean",
    "generation_gwh_rolling_3y_std",
    "generation_gwh_expanding_mean",
    "generation_gwh_vs_company_mean",
    "generation_gwh_company_zscore",
    "generation_cumulative_gwh",
    "generation_per_mm_rainfall",
    "generation_to_rainfall_index",
    "Annual Rainfall (mm)",
    "rainfall_mm_lag_1y",
    "rainfall_mm_lag_2y",
    "rainfall_mm_yoy_change",
    "rainfall_mm_yoy_pct_change",
    "rainfall_mm_rolling_3y_mean",
    "rainfall_mm_rolling_3y_std",
    "rainfall_mm_expanding_mean",
    "rainfall_mm_vs_company_mean",
    "rainfall_mm_company_zscore",
    "rainfall_cumulative_mm",
    "rainfall_to_generation_index",
    "monsoon_season",
    "dry_season",
    "month_of_year_sin",
    "month_of_year_cos",
    "rainfall_year_progress",
]

EXCLUDED_HYDRO_FEATURE_COLUMNS = {
    "generation_gwh_lag_1y",
    "generation_gwh_lag_2y",
    "rainfall_mm_lag_1y",
    "rainfall_mm_lag_2y",
    "generation_gwh_rolling_3y_std",
    "rainfall_mm_rolling_3y_std",
}


def configure_target_horizon(horizon: int) -> None:
    if horizon <= 0:
        raise ValueError("target horizon must be a positive integer.")

    global TARGET_HORIZON, TARGET_COL, TARGET_LOG_COL, TARGET_DIRECTION_COL
    TARGET_HORIZON = horizon
    TARGET_COL = f"target_return_{horizon}d"
    TARGET_LOG_COL = f"target_log_return_{horizon}d"
    TARGET_DIRECTION_COL = f"target_direction_{horizon}d"


def set_reproducibility(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ModuleNotFoundError:
        pass


def load_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras import Model, Sequential
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.layers import (
            LSTM,
            Dense,
            Dropout,
            GlobalAveragePooling1D,
            Input,
            LayerNormalization,
        )
        from tensorflow.keras.optimizers import Adam
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required for training. Use the project virtualenv "
            "`venv\\Scripts\\python.exe hybrid_hydro_market_model.py`, or install "
            "requirements with `pip install -r requirements.txt`."
        ) from exc

    return {
        "tf": tf,
        "Model": Model,
        "Sequential": Sequential,
        "EarlyStopping": EarlyStopping,
        "ReduceLROnPlateau": ReduceLROnPlateau,
        "LSTM": LSTM,
        "Dense": Dense,
        "Dropout": Dropout,
        "GlobalAveragePooling1D": GlobalAveragePooling1D,
        "Input": Input,
        "LayerNormalization": LayerNormalization,
        "Adam": Adam,
    }


def ensure_dirs() -> None:
    FEATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_external_features() -> pd.DataFrame:
    production = pd.read_csv(PRODUCTION_FEATURE_PATH)
    rainfall = pd.read_csv(RAINFALL_FEATURE_PATH)

    production = production.drop(
        columns=[
            "Station Location",
            "Rainfall Station (District, Elevation)",
        ],
        errors="ignore",
    )
    rainfall = rainfall.drop(
        columns=[
            "Station Location",
            "Rainfall Station (District, Elevation)",
            "rainfall_intensity_category",
        ],
        errors="ignore",
    )

    external = production.merge(rainfall, on=["Company", "Year"], how="inner")
    external["Company"] = external["Company"].astype(str).str.upper()
    external["Year"] = pd.to_numeric(external["Year"], errors="coerce").astype(int)
    return external


def load_market_features(company: str) -> pd.DataFrame:
    path = PRICE_FEATURE_DIR / f"{company}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Market feature file not found: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df.sort_index()
    df["Company"] = company
    df["Year"] = df.index.year

    if (
        TARGET_COL not in df.columns
        or TARGET_LOG_COL not in df.columns
        or TARGET_DIRECTION_COL not in df.columns
    ):
        future = df["close"].shift(-TARGET_HORIZON)
        df[TARGET_COL] = (future - df["close"]) / df["close"]
        df[TARGET_LOG_COL] = np.log(future / df["close"])
        df[TARGET_DIRECTION_COL] = np.where(
            future.notna(), (future > df["close"]).astype(int), np.nan
        )

    return df


def add_daily_seasonal_hydro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily seasonal features to a hydrology-market dataframe.

    The dataframe must have a DateTimeIndex and should already include the
    annual columns `Energy Generation (GWh)` and `Annual Rainfall (mm)`.
    Original annual columns are kept unchanged.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("add_daily_seasonal_hydro_features requires a DateTimeIndex.")

    out = df.copy()
    month = out.index.month
    day_of_year = out.index.dayofyear
    days_in_year = np.where(out.index.is_leap_year, 366, 365)

    out["monsoon_season"] = month.isin([6, 7, 8, 9]).astype(int)
    out["dry_season"] = month.isin([11, 12, 1, 2, 3]).astype(int)
    out["month_of_year_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_of_year_cos"] = np.cos(2 * np.pi * month / 12)
    out["generation_per_trading_day"] = out["Energy Generation (GWh)"] / 261
    out["rainfall_year_progress"] = day_of_year / days_in_year
    return out


def impute_hydro_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaN in hydro lag columns with the expanding historical mean
    (computed from available non-NaN values before each row).
    Marks imputed rows with a binary flag column `hydro_lag_imputed`.
    Never uses zero-fill — zero creates a false "no production" signal.
    """
    lag_cols = [c for c in df.columns if "_lag_" in c and
                any(k in c for k in ["generation", "rainfall"])]
    rolling_cols = [c for c in df.columns if "_rolling_" in c or "_expanding_" in c
                    and any(k in c for k in ["generation", "rainfall"])]
    impute_cols = [
        c
        for c in lag_cols + [c for c in rolling_cols if c in df.columns]
        if c not in EXCLUDED_HYDRO_FEATURE_COLUMNS
    ]

    imputed_flag = pd.Series(False, index=df.index)
    out = df.copy()
    for col in impute_cols:
        if col not in out.columns:
            continue
        mask = out[col].isna()
        if mask.any():
            expanding_mean = out[col].expanding().mean()
            out.loc[mask, col] = expanding_mean[mask]
            imputed_flag |= mask

    out["hydro_lag_imputed"] = imputed_flag.astype(int)

    # Print before/after NaN report
    before = df[impute_cols].isnull().sum()
    after  = out[impute_cols].isnull().sum()
    remaining = after[after > 0]
    if not remaining.empty:
        print(f"  [IMPUTE] Warning — still NaN after expanding-mean imputation:")
        print(remaining)
    else:
        total_imputed = int(imputed_flag.sum())
        print(f"  [IMPUTE] {total_imputed} rows imputed across {len(impute_cols)} lag/rolling cols")

    return out


def merge_hydro_market(company: str, external: pd.DataFrame) -> pd.DataFrame:
    market = load_market_features(company)
    merged = market.reset_index().merge(
        external,
        on=["Company", "Year"],
        how="inner",
        validate="many_to_one",
    )
    merged = merged.set_index("Date").sort_index()
    merged = add_daily_seasonal_hydro_features(merged)

    # ── Impute hydro lag NaNs with expanding mean (not zero) ─────────────
    merged = impute_hydro_lags(merged)

    merged = merged.dropna(subset=[TARGET_COL])
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged.to_csv(FEATURE_OUTPUT_DIR / f"{company}_hydro_market_features.csv")
    return merged


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {
        "Company",
        "Year",
        TARGET_COL,
        TARGET_LOG_COL,
        TARGET_DIRECTION_COL,
    }
    skip.update(EXCLUDED_HYDRO_FEATURE_COLUMNS)
    feature_cols = []
    for col in df.columns:
        if col in skip or col.startswith("target_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def split_time_series(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    train_end = int(n_rows * TRAIN_RATIO)
    val_end = int(n_rows * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def make_sequences(
    x_values: np.ndarray, y_values: np.ndarray, seq_length: int
) -> tuple[np.ndarray, np.ndarray]:
    x_sequences, y_sequences = [], []
    for i in range(seq_length, len(x_values)):
        x_sequences.append(x_values[i - seq_length : i])
        y_sequences.append(y_values[i])
    return np.asarray(x_sequences, dtype=np.float32), np.asarray(y_sequences, dtype=np.float32)


def prepare_arrays(
    df: pd.DataFrame, feature_cols: list[str], seq_length: int
) -> dict[str, np.ndarray | list[str]]:
    train, val, test = split_time_series(df)

    # ── Feature scaler (fit on train only) ───────────────────────────────
    feature_scaler = RobustScaler()

    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    x_train = feature_scaler.fit_transform(clean(train))
    x_val   = feature_scaler.transform(clean(val))
    x_test  = feature_scaler.transform(clean(test))

    # ── Raw (unscaled) targets — stored for correct metric computation ───
    y_train_raw = train[TARGET_COL].to_numpy(dtype=np.float32)
    y_val_raw   = val[TARGET_COL].to_numpy(dtype=np.float32)
    y_test_raw  = test[TARGET_COL].to_numpy(dtype=np.float32)

    # ── Target scaler — separate from feature scaler ─────────────────────
    # Scaling the target helps the sequence model converge; inverse-transform before metrics.
    target_scaler = RobustScaler()
    y_train = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten().astype(np.float32)
    y_val   = target_scaler.transform(y_val_raw.reshape(-1, 1)).flatten().astype(np.float32)
    y_test  = target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten().astype(np.float32)

    xs_train, ys_train = make_sequences(x_train, y_train, seq_length)
    xs_val,   ys_val   = make_sequences(x_val,   y_val,   seq_length)
    xs_test,  ys_test  = make_sequences(x_test,  y_test,  seq_length)

    # Persistence baseline: predict future horizon return from the most recent
    # trailing return over the same horizon, aligned with sequence targets.
    persistence_col = f"return_{TARGET_HORIZON}d"
    if persistence_col in test.columns:
        persistence_prediction_raw = (
            test[persistence_col]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
    else:
        persistence_prediction_raw = np.zeros_like(y_test_raw)

    persistence_prediction_raw = persistence_prediction_raw[
        seq_length : seq_length + len(ys_test)
    ]
    min_len = min(len(ys_test), len(persistence_prediction_raw))

    return {
        "X_train": x_train,
        "X_val":   x_val,
        "X_test":  x_test,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "y_train_raw":  y_train_raw,
        "y_test_raw":   y_test_raw,
        "Xs_train": xs_train,
        "Xs_val":   xs_val,
        "Xs_test":  xs_test,
        "ys_train": ys_train,
        "ys_val":   ys_val,
        "ys_test":  ys_test,
        "feature_cols":         feature_cols,
        "target_scaler":        target_scaler,
        "persistence_prediction_raw": persistence_prediction_raw[:min_len],
        "ys_test_aligned":      ys_test[:min_len],
        # fraction of UP days in training set — used for class weighting
        "up_ratio": float((y_train_raw > 0).mean()),
    }


def save_arrays(company: str, arrays: dict[str, np.ndarray | list[str]]) -> None:
    path = STUDY_OUTPUT_DIR / f"{company}_hydro_market_arrays.npz"
    np.savez(
        path,
        X_train=arrays["X_train"],
        X_val=arrays["X_val"],
        X_test=arrays["X_test"],
        y_train=arrays["y_train"],
        y_val=arrays["y_val"],
        y_test=arrays["y_test"],
        y_train_raw=arrays["y_train_raw"],
        y_test_raw=arrays["y_test_raw"],
        Xs_train=arrays["Xs_train"],
        Xs_val=arrays["Xs_val"],
        Xs_test=arrays["Xs_test"],
        ys_train=arrays["ys_train"],
        ys_val=arrays["ys_val"],
        ys_test=arrays["ys_test"],
        persistence_prediction_raw=arrays["persistence_prediction_raw"],
        feature_cols=np.asarray(arrays["feature_cols"]),
    )

    with open(STUDY_OUTPUT_DIR / f"{company}_feature_groups.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": TARGET_COL,
                "target_horizon_days": TARGET_HORIZON,
                "sequence_length": SEQ_LENGTH,
                "hydro_production_features": [
                    col for col in arrays["feature_cols"] if col in HYDRO_PRODUCTION_COLUMNS
                ],
                "market_features": [
                    col for col in arrays["feature_cols"] if col not in HYDRO_PRODUCTION_COLUMNS
                ],
            },
            f,
            indent=2,
        )


def build_lstm(input_shape: tuple[int, int], learning_rate: float, keras_api):
    Input = keras_api["Input"]
    LSTM = keras_api["LSTM"]
    Dropout = keras_api["Dropout"]
    GlobalAveragePooling1D = keras_api["GlobalAveragePooling1D"]
    LayerNormalization = keras_api["LayerNormalization"]
    Dense = keras_api["Dense"]
    Sequential = keras_api["Sequential"]
    Adam = keras_api["Adam"]

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(
                48,
                return_sequences=True,
                dropout=0.20,
                recurrent_dropout=0.10,
            ),
            LayerNormalization(),
            Dropout(0.30),
            LSTM(
                24,
                return_sequences=True,
                dropout=0.20,
                recurrent_dropout=0.10,
            ),
            GlobalAveragePooling1D(),
            Dense(16, activation="relu"),
            Dropout(0.15),
            Dense(1, name=f"next_{TARGET_HORIZON}d_return"),
        ],
        name="Regularized_LSTM",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="huber",
        metrics=["mae"],
    )
    return model


def evaluate_predictions(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    target_scaler: RobustScaler | None = None,
) -> dict[str, float]:
    """
    Compute RMSE, MAE, R², and directional accuracy.

    If target_scaler is supplied both y_true and y_pred are inverse-transformed
    to the original return scale before computing any metric.  This is the only
    correct way to get a meaningful R² when the target was scaled.
    """
    y_pred_scaled = y_pred_scaled.reshape(-1)

    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(
            y_true_scaled.reshape(-1, 1)
        ).flatten()
        y_pred = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()
    else:
        y_true = y_true_scaled
        y_pred = y_pred_scaled

    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    dir_acc = float(accuracy_score(y_true > 0, y_pred > 0))
    return {
        "rmse": rmse,
        "mae":  mae,
        "r2":   r2,
        "directional_accuracy": dir_acc,
    }


def evaluate_persistence_baseline(
    arrays: dict,
) -> dict[str, float]:
    """Naive baseline: predict future return from recent same-horizon return."""
    y_pred_raw = arrays["persistence_prediction_raw"]
    ys_aligned = arrays["ys_test_aligned"]
    target_scaler = arrays["target_scaler"]

    y_pred_scaled = target_scaler.transform(y_pred_raw.reshape(-1, 1)).flatten()
    min_len = min(len(ys_aligned), len(y_pred_scaled))
    metrics = evaluate_predictions(
        ys_aligned[:min_len], y_pred_scaled[:min_len], target_scaler
    )
    metrics["model"]   = "Persistence_Baseline"
    metrics["company"] = ""
    return metrics


def train_one_model(
    model_name: str,
    model,
    arrays: dict[str, np.ndarray | list[str]],
    epochs: int,
    batch_size: int,
) -> tuple[dict[str, float], object, np.ndarray]:
    keras_api = load_tensorflow()
    EarlyStopping    = keras_api["EarlyStopping"]
    ReduceLROnPlateau= keras_api["ReduceLROnPlateau"]

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    history = model.fit(
        arrays["Xs_train"],
        arrays["ys_train"],
        validation_data=(arrays["Xs_val"], arrays["ys_val"]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )
    predictions_scaled = model.predict(arrays["Xs_test"], verbose=0).reshape(-1)

    # ── Inverse-transform before computing metrics ───────────────────────
    target_scaler = arrays.get("target_scaler")
    metrics = evaluate_predictions(arrays["ys_test"], predictions_scaled, target_scaler)
    metrics["model"] = model_name

    # Store real-scale predictions for plotting
    if target_scaler is not None:
        predictions_real = target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
    else:
        predictions_real = predictions_scaled

    return metrics, history, predictions_real


def plot_training_history(company: str, model_name: str, history: object) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history.history["loss"], label="Train loss")
    ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set_title(f"{company} {model_name} Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(STUDY_OUTPUT_DIR / f"{company}_{model_name}_training_loss.png")
    plt.close(fig)


def plot_predictions(
    company: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(y_true, label=f"Actual {TARGET_HORIZON}-day return", linewidth=1.8)
    ax.plot(
        y_pred,
        label=f"Predicted {TARGET_HORIZON}-day return",
        linewidth=1.4,
        alpha=0.85,
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"{company} {model_name}: Actual vs Predicted Market Movement")
    ax.set_xlabel("Test sequence")
    ax.set_ylabel("Return")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(STUDY_OUTPUT_DIR / f"{company}_{model_name}_predictions.png")
    plt.close(fig)


def train_gradient_boosting(
    company: str,
    arrays: dict[str, np.ndarray | list[str]],
) -> dict[str, float | str]:
    """
    GradientBoostingRegressor with class-imbalance-aware sample weights.
    Up-days and down-days are weighted inversely proportional to their frequency
    so minority-class samples count more during training.
    """
    up_ratio   = arrays.get("up_ratio", 0.5)
    down_ratio = 1.0 - up_ratio

    # sample_weight: give minority class higher weight
    y_train_raw = arrays["y_train_raw"]
    if up_ratio < 0.5:
        # more DOWN days — upweight UP days
        w_up, w_down = down_ratio / up_ratio, 1.0
    else:
        # more UP days — upweight DOWN days
        w_up, w_down = 1.0, up_ratio / down_ratio

    sample_weights = np.where(y_train_raw > 0, w_up, w_down).astype(np.float32)

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=8,
        subsample=0.80,
        random_state=RANDOM_SEED,
    )
    model.fit(arrays["X_train"], y_train_raw, sample_weight=sample_weights)

    # Predict on raw (unscaled) test targets — GB works in original space
    y_test_raw   = arrays["y_test_raw"]
    predictions  = model.predict(arrays["X_test"])

    # Align length: X_test has more rows than ys_test (no sequence offset needed for GB)
    min_len = min(len(y_test_raw), len(predictions))
    metrics = evaluate_predictions(
        y_test_raw[:min_len], predictions[:min_len], target_scaler=None
    )
    metrics["company"] = company
    metrics["model"]   = "GradientBoosting"

    joblib.dump(model, STUDY_OUTPUT_DIR / f"{company}_GradientBoosting.joblib")
    plot_predictions(company, "GradientBoosting", y_test_raw[:min_len], predictions[:min_len])
    return metrics


def train_xgboost(
    company: str,
    arrays: dict[str, np.ndarray | list[str]],
) -> dict[str, float | str]:
    """
    XGBoost regressor with class-imbalance-aware sample weights.
    """
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "XGBoost is required for the XGBoost model. Use the project virtualenv "
            "`venv\\Scripts\\python.exe hybrid_hydro_market_model.py`, or install "
            "requirements with `pip install -r requirements.txt`."
        ) from exc

    up_ratio = arrays.get("up_ratio", 0.5)
    down_ratio = 1.0 - up_ratio
    y_train_raw = arrays["y_train_raw"]

    if up_ratio < 0.5:
        w_up, w_down = down_ratio / up_ratio, 1.0
    else:
        w_up, w_down = 1.0, up_ratio / down_ratio

    sample_weights = np.where(y_train_raw > 0, w_up, w_down).astype(np.float32)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=8,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_lambda=5.0,
        reg_alpha=0.10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(arrays["X_train"], y_train_raw, sample_weight=sample_weights)

    y_test_raw = arrays["y_test_raw"]
    predictions = model.predict(arrays["X_test"])

    min_len = min(len(y_test_raw), len(predictions))
    metrics = evaluate_predictions(
        y_test_raw[:min_len], predictions[:min_len], target_scaler=None
    )
    metrics["company"] = company
    metrics["model"] = "XGBoost"

    joblib.dump(model, STUDY_OUTPUT_DIR / f"{company}_XGBoost.joblib")
    plot_predictions(company, "XGBoost", y_test_raw[:min_len], predictions[:min_len])
    return metrics


def relation_summary(company: str, df: pd.DataFrame) -> list[dict[str, float | str]]:
    numeric_cols = [
        col
        for col in HYDRO_PRODUCTION_COLUMNS
        if (
            col in df.columns
            and col not in EXCLUDED_HYDRO_FEATURE_COLUMNS
            and pd.api.types.is_numeric_dtype(df[col])
        )
    ]
    rows = []
    for col in numeric_cols:
        valid = df[[col, TARGET_COL]].dropna()
        if len(valid) < 3 or valid[col].nunique() <= 1:
            continue
        rows.append(
            {
                "company": company,
                "feature": col,
                "corr_with_target_return": float(valid[col].corr(valid[TARGET_COL])),
                "mean_value": float(valid[col].mean()),
                "std_value": float(valid[col].std()),
            }
        )
    return rows


def plot_relation_bars(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.copy()
    plot_df["abs_corr"] = plot_df["corr_with_target_return"].abs()
    plot_df = plot_df.sort_values("abs_corr", ascending=False).head(20)

    labels = plot_df["company"] + ": " + plot_df["feature"]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = np.where(plot_df["corr_with_target_return"] >= 0, "#1E8449", "#922B21")
    ax.barh(labels, plot_df["corr_with_target_return"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_title(
        f"Hydrology and Production Correlation with {TARGET_HORIZON}-Day Market Return"
    )
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(STUDY_OUTPUT_DIR / "hydro_production_target_correlations.png")
    plt.close(fig)


def train_models_for_company(
    company: str,
    arrays: dict[str, np.ndarray | list[str]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> list[dict[str, float | str]]:
    keras_api = load_tensorflow()
    input_shape = arrays["Xs_train"].shape[1:]
    if arrays["Xs_train"].shape[0] == 0 or arrays["Xs_val"].shape[0] == 0:
        raise ValueError(f"{company} does not have enough rows for sequence training.")

    results = []

    # ── Persistence baseline ──────────────────────────────────────────────
    baseline_metrics = evaluate_persistence_baseline(arrays)
    baseline_metrics["company"] = company
    results.append(baseline_metrics)
    print(
        f"  [Baseline] RMSE={baseline_metrics['rmse']:.4f} "
        f"MAE={baseline_metrics['mae']:.4f} "
        f"R²={baseline_metrics['r2']:.4f} "
        f"DirAcc={baseline_metrics['directional_accuracy']:.4f}"
    )

    # Sequence model
    model_name = "Regularized_LSTM"
    model = build_lstm(input_shape, learning_rate, keras_api)
    metrics, history, predictions_real = train_one_model(
        model_name, model, arrays, epochs=epochs, batch_size=batch_size,
    )
    metrics["company"] = company
    results.append(metrics)
    model.save(STUDY_OUTPUT_DIR / f"{company}_{model_name}.keras")
    plot_training_history(company, model_name, history)

    # Plot uses real-scale test targets
    target_scaler = arrays.get("target_scaler")
    y_test_real = (
        target_scaler.inverse_transform(arrays["ys_test"].reshape(-1, 1)).flatten()
        if target_scaler else arrays["ys_test"]
    )
    plot_predictions(company, model_name, y_test_real, predictions_real)

    # ── Gradient Boosting ─────────────────────────────────────────────────
    results.append(train_gradient_boosting(company, arrays))
    results.append(train_xgboost(company, arrays))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Regularized LSTM, Gradient Boosting, and XGBoost models with rainfall, production, and market features."
    )
    parser.add_argument("--companies", nargs="+", default=COMPANIES, choices=COMPANIES)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--sequence-length", type=int, default=SEQ_LENGTH)
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=DEFAULT_TARGET_HORIZON,
        help="Future return horizon in trading days, e.g. 5, 20, or 60.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only create merged features and arrays; skip TensorFlow training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_target_horizon(args.target_horizon)
    set_reproducibility()
    ensure_dirs()

    external = load_external_features()
    all_results = []
    relation_rows = []

    for company in args.companies:
        print(f"\n[{company}] Merging hydrology, production, and market features")
        merged = merge_hydro_market(company, external)
        feature_cols = get_feature_columns(merged)
        hydro_cols = [col for col in feature_cols if col in HYDRO_PRODUCTION_COLUMNS]
        print(
            f"  rows={len(merged)} | features={len(feature_cols)} "
            f"| hydro/production features={len(hydro_cols)}"
        )

        arrays = prepare_arrays(merged, feature_cols, args.sequence_length)
        save_arrays(company, arrays)
        relation_rows.extend(relation_summary(company, merged))

        if args.dry_run:
            continue

        print(f"[{company}] Training Regularized LSTM, Gradient Boosting, and XGBoost")
        all_results.extend(
            train_models_for_company(
                company,
                arrays,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
        )

    relation_df = pd.DataFrame(relation_rows)
    relation_df.to_csv(STUDY_OUTPUT_DIR / "hydro_market_relation_summary.csv", index=False)
    plot_relation_bars(relation_df)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df[
            ["company", "model", "rmse", "mae", "r2", "directional_accuracy"]
        ].sort_values(["company", "rmse"])
        results_df.to_csv(STUDY_OUTPUT_DIR / "hydro_market_model_results.csv", index=False)
        print("\nModel results")
        print(results_df.to_string(index=False))
    else:
        print("\nDry run complete. Merged features, arrays, and relation summary were created.")

    print(f"\nSaved study outputs in: {STUDY_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
