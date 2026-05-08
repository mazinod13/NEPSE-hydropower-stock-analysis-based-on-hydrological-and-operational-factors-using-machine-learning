# NEPSE Hydropower ML Study Documentation

This document summarizes the work completed so far for studying the relationship between hydrological factors, production rate, and market movement for NEPSE hydropower companies.

## 1. Project Goal

The current pipeline prepares data and models for studying whether annual hydrological and operational factors help explain or predict short-term market movement in hydropower stocks.

The study combines three data groups:

- Market data: daily OHLCV and technical indicators from `Output/{COMPANY}_features.csv`
- Production data: annual energy generation in GWh from `Datasets/rainfall&production.csv`
- Hydrological data: annual rainfall in mm from `Datasets/rainfall&production.csv`

The prediction target is configurable by trading-day horizon:

```text
target_return_{horizon}d = future horizon-trading-day stock return
```

The default horizon is 5 trading days, but the model can also be run with longer horizons such as 20 or 60 trading days.

Models prepared:

- Regularized LSTM
- Gradient Boosting Regressor
- XGBoost Regressor

## 2. Files Created

### 2.1 Rainfall and Production Preprocessing

Script:

```text
rainfall_production_preprocessing.py
```

Input:

```text
Datasets/rainfall&production.csv
```

Outputs:

```text
Features/Production/production_features.csv
Features/Production/{COMPANY}_production_features.csv
Features/Rainfall/rainfall_features.csv
Features/Rainfall/{COMPANY}_rainfall_features.csv
Output/RP_01_production_trends.png
Output/RP_02_rainfall_trends.png
Output/RP_03_rainfall_vs_production.png
Output/RP_04_company_generation_rainfall.png
Output/RP_05_feature_correlation.png
```

Purpose:

- Load the annual rainfall and production dataset
- Fix the malformed CSV header caused by the comma inside `Rainfall Station (District, Elevation)`
- Engineer meaningful hydrology and production features
- Save combined and per-company feature files
- Generate charts for rainfall, production, and their relationship

### 2.2 Hybrid Hydrology-Market Modeling

Script:

```text
hybrid_hydro_market_model.py
```

Inputs:

```text
Output/BARUN_features.csv
Output/CHCL_features.csv
Output/RADHI_features.csv
Features/Production/production_features.csv
Features/Rainfall/rainfall_features.csv
```

Outputs:

```text
Features/HydroMarket/{COMPANY}_hydro_market_features.csv
Output/HydroMarketStudy/{COMPANY}_hydro_market_arrays.npz
Output/HydroMarketStudy/{COMPANY}_feature_groups.json
Output/HydroMarketStudy/hydro_market_relation_summary_{COMPANY}.csv
Output/HydroMarketStudy/hydro_production_target_correlations_{COMPANY}.png
Output/HydroMarketStudy/hydro_market_model_results_{COMPANY}.csv
Output/HydroMarketStudy/hydro_market_model_results_ALL.csv
Output/HydroMarketStudy/{COMPANY}_Regularized_LSTM.keras
Output/HydroMarketStudy/{COMPANY}_GradientBoosting.joblib
Output/HydroMarketStudy/{COMPANY}_XGBoost.joblib
Output/HydroMarketStudy/{COMPANY}_Regularized_LSTM_training_loss.png
Output/HydroMarketStudy/{COMPANY}_Regularized_LSTM_predictions.png
Output/HydroMarketStudy/{COMPANY}_GradientBoosting_predictions.png
Output/HydroMarketStudy/{COMPANY}_XGBoost_predictions.png
```

Purpose:

- Merge annual rainfall and production features into daily market features by `Company + Year`
- Create scaled sequence arrays for TensorFlow models
- Train and compare Regularized LSTM, Gradient Boosting, and XGBoost models
- Evaluate model performance
- Save model files, predictions charts, training charts, and result summaries

## 3. Dataset Structure

The rainfall and production dataset contains:

```text
Company
Station Location
Rainfall Station (District, Elevation)
Year
Energy Generation (GWh)
Annual Rainfall (mm)
```

Companies currently included:

```text
BARUN
CHCL
RADHI
```

Years currently included:

```text
2018 to 2025
```

## 4. Production Feature Formulas

Let:

```text
G_t = Energy Generation (GWh) in year t
R_t = Annual Rainfall (mm) in year t
```

### 4.1 Lag Features

Previous year's generation:

```text
generation_gwh_lag_1y = G_{t-1}
```

Generation from two years ago:

```text
generation_gwh_lag_2y = G_{t-2}
```

Purpose:

- Captures delayed or historical production behavior
- Helps the model compare current market movement with previous operational performance

### 4.2 Year-over-Year Change

Absolute change:

```text
generation_gwh_yoy_change = G_t - G_{t-1}
```

Percentage change:

```text
generation_gwh_yoy_pct_change = (G_t - G_{t-1}) / G_{t-1}
```

Purpose:

- Measures whether production improved or declined compared to the previous year

### 4.3 Rolling Mean

Three-year rolling average:

```text
generation_gwh_rolling_3y_mean = mean(G_t, G_{t-1}, G_{t-2})
```

Purpose:

- Smooths short-term variation
- Represents recent production trend

### 4.4 Rolling Standard Deviation

Three-year rolling volatility:

```text
generation_gwh_rolling_3y_std = std(G_t, G_{t-1}, G_{t-2})
```

Purpose:

- Measures production stability or variability

### 4.5 Expanding Mean

Average production from the first available year up to year `t`:

```text
generation_gwh_expanding_mean = mean(G_1, G_2, ..., G_t)
```

Purpose:

- Captures long-term company-level production behavior

### 4.6 Deviation from Company Mean

```text
generation_gwh_vs_company_mean = G_t - mean(G)
```

Purpose:

- Shows whether the current year is above or below the company's normal production level

### 4.7 Company Z-Score

```text
generation_gwh_company_zscore = (G_t - mean(G)) / std(G)
```

Purpose:

- Standardizes production within each company
- Makes production anomalies comparable across companies

### 4.8 Cumulative Generation

```text
generation_cumulative_gwh = G_1 + G_2 + ... + G_t
```

Purpose:

- Represents total recorded energy generation over time

### 4.9 Generation per Rainfall

```text
generation_per_mm_rainfall = G_t / R_t
```

Purpose:

- Measures how much electricity was generated per millimeter of annual rainfall
- Acts as a rough operational efficiency indicator

### 4.10 Generation-to-Rainfall Index

```text
generation_to_rainfall_index =
    generation_per_mm_rainfall / mean(generation_per_mm_rainfall)
```

Purpose:

- Compares a year's rainfall-production efficiency against the company's average
- Value above `1` means better than usual generation per rainfall
- Value below `1` means weaker than usual generation per rainfall

## 5. Rainfall Feature Formulas

Let:

```text
R_t = Annual Rainfall (mm) in year t
```

### 5.1 Lag Features

```text
rainfall_mm_lag_1y = R_{t-1}
rainfall_mm_lag_2y = R_{t-2}
```

Purpose:

- Captures previous hydrological conditions

### 5.2 Year-over-Year Rainfall Change

Absolute change:

```text
rainfall_mm_yoy_change = R_t - R_{t-1}
```

Percentage change:

```text
rainfall_mm_yoy_pct_change = (R_t - R_{t-1}) / R_{t-1}
```

Purpose:

- Measures whether rainfall increased or decreased compared with the previous year

### 5.3 Rolling Mean

```text
rainfall_mm_rolling_3y_mean = mean(R_t, R_{t-1}, R_{t-2})
```

Purpose:

- Represents recent hydrological trend

### 5.4 Rolling Standard Deviation

```text
rainfall_mm_rolling_3y_std = std(R_t, R_{t-1}, R_{t-2})
```

Purpose:

- Measures rainfall variability

### 5.5 Expanding Mean

```text
rainfall_mm_expanding_mean = mean(R_1, R_2, ..., R_t)
```

Purpose:

- Represents long-term average rainfall for each company station

### 5.6 Deviation from Company Mean

```text
rainfall_mm_vs_company_mean = R_t - mean(R)
```

Purpose:

- Shows whether rainfall is above or below the station's normal level

### 5.7 Rainfall Z-Score

```text
rainfall_mm_company_zscore = (R_t - mean(R)) / std(R)
```

Purpose:

- Standardizes rainfall within each company/station
- Useful for identifying unusually dry or wet years

### 5.8 Cumulative Rainfall

```text
rainfall_cumulative_mm = R_1 + R_2 + ... + R_t
```

Purpose:

- Tracks accumulated rainfall over the available period

### 5.9 Rainfall Intensity Category

The rainfall values are ranked within each company and divided into three groups:

```text
low
normal
high
```

Purpose:

- Provides an easy categorical interpretation of rainfall intensity

### 5.10 Rainfall-to-Generation Index

```text
rainfall_to_generation_index = R_t / mean(R)
```

Purpose:

- Compares annual rainfall against the company's average rainfall
- Value above `1` means wetter than average
- Value below `1` means drier than average

### 5.11 Daily Seasonal Features Added After Annual Merge

The original rainfall and production values are annual. When they are merged into the daily stock market dataset, the same annual value repeats across all trading days in that year.

This is kept as yearly context, but additional daily seasonal features are added so the model receives more useful time-varying hydrological signals.

These features are added in `hybrid_hydro_market_model.py` after merging annual hydrology-production features with the daily market dataframe.

The dataframe must already have a `DateTimeIndex`.

```python
def add_daily_seasonal_hydro_features(df):
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
```

Monsoon season flag:

```text
monsoon_season = 1 if month is June, July, August, or September
monsoon_season = 0 otherwise
```

Purpose:

- Captures Nepal's main rainfall season
- Helps the model distinguish wet-season trading periods from the rest of the year

Dry season flag:

```text
dry_season = 1 if month is November, December, January, February, or March
dry_season = 0 otherwise
```

Purpose:

- Captures low-rainfall months
- Helps the model identify periods where hydropower generation pressure may be different

Month cyclic encoding:

```text
month_of_year_sin = sin(2 * pi * month / 12)
month_of_year_cos = cos(2 * pi * month / 12)
```

Purpose:

- Represents month as a cycle rather than a straight number
- Makes December and January close to each other mathematically
- Helps the model learn annual seasonality

Generation per trading day:

```text
generation_per_trading_day = annual_generation_gwh / 261
```

Purpose:

- Converts annual generation into an approximate daily market-period estimate
- Uses 261 as a common estimate for yearly trading days
- Keeps the original annual generation column unchanged

Rainfall year progress:

```text
rainfall_year_progress = day_of_year / days_in_year
```

Purpose:

- Shows how far through the year the current trading day is
- Helps the model distinguish early-year, mid-year, and late-year use of the same annual rainfall value
- Uses 366 days for leap years and 365 days otherwise

## 6. Market Feature Terms Used

The existing market feature files in `Output/{COMPANY}_features.csv` already include technical market indicators.

Common terms include:

### 6.1 OHLCV

```text
open   = opening price
high   = highest price of the day
low    = lowest price of the day
close  = closing price
volume = traded shares/units
```

### 6.2 Daily Return

```text
return_1d = (close_t - close_{t-1}) / close_{t-1}
```

For longer windows:

```text
return_nd = (close_t - close_{t-n}) / close_{t-n}
```

Examples:

```text
return_2d
return_3d
return_5d
return_10d
return_20d
```

### 6.3 Log Return

```text
log_return_1d = ln(close_t / close_{t-1})
```

Purpose:

- Often preferred in financial modeling because log returns are additive over time

### 6.4 Overnight Gap

```text
overnight_gap = (open_t - close_{t-1}) / close_{t-1}
```

Purpose:

- Measures price movement between previous close and current open

### 6.5 Intraday Range

```text
intraday_range = (high_t - low_t) / close_t
```

Purpose:

- Measures daily price volatility

### 6.6 Close-Open Return

```text
close_open_return = (close_t - open_t) / open_t
```

Purpose:

- Measures price movement during the trading day

### 6.7 Moving Average

```text
ma_n = mean(close_t, close_{t-1}, ..., close_{t-n+1})
```

Examples:

```text
ma_5
ma_10
ma_20
ma_50
```

Purpose:

- Smooths market price movement
- Helps detect trend direction

### 6.8 Rolling Standard Deviation

```text
std_n = standard deviation of closing price over n days
```

Purpose:

- Measures price volatility

### 6.9 Price Relative to Moving Average

```text
price_rel_ma_n = close_t / ma_n
```

Purpose:

- Value above `1` means price is above its moving average
- Value below `1` means price is below its moving average

### 6.10 Volume Moving Average

```text
vol_ma_n = mean(volume_t, volume_{t-1}, ..., volume_{t-n+1})
```

Purpose:

- Measures normal trading activity over a rolling window

### 6.11 Volume Surge

```text
vol_surge_20 = volume_t / vol_ma_20
```

Purpose:

- Detects unusually high or low trading activity

### 6.12 MACD

MACD is based on exponential moving averages:

```text
MACD = EMA_12(close) - EMA_26(close)
MACD Signal = EMA_9(MACD)
MACD Diff = MACD - MACD Signal
```

Purpose:

- Momentum indicator
- Helps detect trend shifts

### 6.13 RSI

Relative Strength Index:

```text
RSI = 100 - (100 / (1 + RS))
RS = average gain / average loss
```

Purpose:

- Momentum oscillator
- Higher RSI can indicate overbought behavior
- Lower RSI can indicate oversold behavior

### 6.14 Bollinger Bands

```text
bb_upper = ma_20 + 2 * std_20
bb_lower = ma_20 - 2 * std_20
bb_width = (bb_upper - bb_lower) / ma_20
bb_pct = (close_t - bb_lower) / (bb_upper - bb_lower)
```

Purpose:

- Measures price position and volatility relative to recent price range

## 7. Target Variable

The main prediction target is future return over a configurable trading-day horizon.

Let:

```text
h = target horizon in trading days
```

Then:

```text
target_return_{h}d = (close_{t+h} - close_t) / close_t
```

The direction target is:

```text
target_direction_{h}d = 1 if close_{t+h} > close_t else 0
```

Purpose:

- `target_return_{h}d` is used for regression
- `target_direction_{h}d` is used to measure market movement direction

Default:

```text
h = 5
```

For hydrology and production studies, longer horizons can be more meaningful:

```text
h = 20   roughly one trading month
h = 60   roughly one trading quarter
```

Reason:

- Rainfall and production are annual or seasonal signals
- Their effect may be too slow to appear in 5 trading days
- Monthly or quarterly horizons can better match hydrological and operational timescales

## 8. Train, Validation, and Test Split

The model uses time-based splitting, not random splitting.

For each company:

```text
First 70% of rows  = training set
Next 15% of rows   = validation set
Final 15% of rows  = test set
```

Formula:

```text
train_end = int(number_of_rows * 0.70)
val_end = int(number_of_rows * 0.85)
```

Purpose:

- Prevents future data from leaking into the past
- More realistic for stock prediction
- The model trains on older data and tests on newer unseen data

## 9. Sequence Creation

The model uses a sequence length of 60 trading days:

```text
SEQ_LENGTH = 60
```

Each model input contains the previous 60 trading days of features.

Example:

```text
Days 1-60 features -> predict Day 61 target_return_{h}d
Days 2-61 features -> predict Day 62 target_return_{h}d
Days 3-62 features -> predict Day 63 target_return_{h}d
```

Purpose:

- Allows LSTM layers to learn temporal patterns
- Captures short-term market memory and trend behavior

## 10. Scaling

The model uses `RobustScaler`.

Formula:

```text
scaled_x = (x - median(x)) / IQR(x)
```

Where:

```text
IQR = Q3 - Q1
```

Purpose:

- Reduces the effect of outliers
- Useful for stock data because market features can have extreme values
- The scaler is fit only on training data, then applied to validation and test data

## 11. Regularized LSTM Model

The Regularized LSTM model replaces the Simple GRU because the GRU was only marginally better than the persistence baseline. LSTM has stronger memory gates and can sometimes capture longer temporal dependencies better than GRU, especially when market movement depends on patterns spread across the full 60-day sequence.

Architecture:

```text
Input: 60 days x feature_count
LSTM(48)
LayerNormalization
Dropout
LSTM(24)
GlobalAveragePooling1D
Dense
Dropout
Output: predicted horizon-day return
```

Purpose of layers:

- `LSTM`: learns sequence dependencies using input, forget, and output gates
- `LayerNormalization`: stabilizes hidden activations during sequence training
- `Dropout`: reduces overfitting
- `GlobalAveragePooling1D`: summarizes sequence-level information
- `Dense`: maps learned patterns to final return prediction

The model uses Huber loss:

```text
loss = huber
```

Purpose:

- Less sensitive to large outlier returns than pure MSE
- Often more stable for noisy financial targets

## 12. Gradient Boosting Model

Gradient Boosting is a tree-based regression model from scikit-learn.

It trains many small decision trees sequentially. Each new tree tries to correct the errors made by the previous trees.

Architecture:

```text
Input: flat daily feature row
Decision tree 1 -> error correction
Decision tree 2 -> error correction
...
Combined trees
Output: predicted horizon-day return
```

Purpose:

- Provides a strong non-deep-learning baseline
- Handles nonlinear feature relationships
- Usually works well on smaller tabular datasets
- Does not need 60-day sequences; it uses the current day's engineered features directly

## 12.1 XGBoost Model

XGBoost is an optimized gradient boosting model designed for strong tabular-data performance.

It is added as a second tree-based model beside scikit-learn Gradient Boosting.

Architecture:

```text
Input: flat daily feature row
Boosted decision trees
Regularization
Column and row subsampling
Output: predicted horizon-day return
```

Purpose:

- Provides a stronger boosted-tree baseline than standard Gradient Boosting
- Handles nonlinear relationships between market, rainfall, and production features
- Includes regularization through `reg_lambda` and `reg_alpha`
- Uses flat daily features, so it is not affected by `--sequence-length`
- Complements LSTM by testing whether tabular relationships outperform sequence learning

## 13. Reason for Changing the Model

The initial modeling approach used a more complex deep learning setup with CNN-LSTM and GRU. During training, the model showed signs of overfitting.

Overfitting means:

```text
Training loss decreases,
but validation loss stops improving or gets worse.
```

This happened because:

- The daily market dataset is not very large after merging with annual hydrology and production data
- Rainfall and production data are annual, so the same yearly values are repeated across many daily rows
- The CNN-LSTM model had more parameters and could learn noise from the training period quickly
- Stock market movement is noisy, so a highly flexible model can memorize historical movement without improving future prediction

Because of this, two improvements were made:

- Static annual columns were kept but supplemented with daily seasonal features
- The model design was changed to a simpler and more controlled comparison

### 13.1 Why LSTM Replaced Simple GRU

Simple GRU was tested because:

- GRU is suitable for time-series sequence learning
- It has fewer parameters than LSTM
- It is usually faster to train
- It can still learn temporal market behavior from 60-day sequences
- The smaller architecture reduces the risk of memorizing training noise

However, the Simple GRU result was very close to the persistence baseline. This suggested that it was not extracting enough additional value from the sequence.

The sequence model was therefore changed to a regularized LSTM:

```text
LSTM(48, dropout=0.20, recurrent_dropout=0.10)
LayerNormalization
Dropout(0.30)
LSTM(24, dropout=0.20, recurrent_dropout=0.10)
GlobalAveragePooling1D
Dense(16)
Dropout(0.15)
Output
```

This keeps the model smaller than the previous CNN-LSTM architecture but gives it more explicit sequence memory than GRU.

### 13.2 Why Gradient Boosting Was Chosen

Gradient Boosting was added as a strong non-deep-learning baseline.

It was chosen because:

- It works well on tabular data
- It can capture nonlinear relationships between rainfall, production, and market features
- It is less dependent on large datasets than deep learning models
- It is easier to interpret and compare
- It does not require sequence construction
- It is already available through scikit-learn in the current project requirements

This gives the study a useful comparison:

```text
Regularized LSTM     = sequence-based deep learning model
Gradient Boosting    = flat-feature machine learning baseline
```

If Gradient Boosting performs better than LSTM, it may suggest that the available hydrology-production-market data is better suited for tabular machine learning than deep sequence modeling.

### 13.3 Why Seasonal Hydro Features Were Added

Annual rainfall and production values alone provide limited daily variation.

Example:

```text
Annual Rainfall (mm) for CHCL in 2018 = 1904.4
```

After merging with daily market data, that same value can repeat for every trading day in 2018.

This creates a problem:

```text
The model sees the annual condition,
but it cannot tell whether the trading day is in monsoon, dry season, or year-end.
```

The added seasonal features solve this by giving the model daily context while preserving the original annual columns.

The new features are:

```text
monsoon_season
dry_season
month_of_year_sin
month_of_year_cos
generation_per_trading_day
rainfall_year_progress
```

This helps the model study hydropower market movement with both:

```text
Annual context      = rainfall and generation level for the year
Daily seasonality   = month, monsoon/dry period, and year progress
```

### 13.4 Why XGBoost Was Added

XGBoost was added after the initial Gradient Boosting experiments because the results showed that tree-based models can be useful for some companies and horizons.

It is now listed in `requirements.txt`:

```text
xgboost==3.2.0
```

The model comparison now includes:

```text
Regularized LSTM
Scikit-learn Gradient Boosting
XGBoost
```

Reason:

- XGBoost is often stronger than standard Gradient Boosting on tabular data
- It supports regularization and subsampling
- It may capture nonlinear feature interactions that LSTM misses
- It gives the thesis a stronger machine-learning baseline against the deep learning model

## 14. Model Loss and Metrics

### 14.1 Training Loss

The model is trained with Mean Squared Error:

```text
MSE = mean((actual - predicted)^2)
```

Purpose:

- Penalizes larger prediction errors more strongly

### 14.2 Mean Absolute Error

```text
MAE = mean(abs(actual - predicted))
```

Purpose:

- Average absolute prediction error
- Easier to interpret than MSE

### 14.3 Root Mean Squared Error

```text
RMSE = sqrt(mean((actual - predicted)^2))
```

Purpose:

- Measures prediction error in the same unit as the target return

### 14.4 R-Squared

```text
R2 = 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
```

Purpose:

- Measures how much variance in the target is explained by the model
- Higher is better
- Negative values mean the model performs worse than predicting the test mean

### 14.5 Directional Accuracy

```text
directional_accuracy =
    count(sign(actual) == sign(predicted)) / total_predictions
```

Purpose:

- Measures whether the model correctly predicts market movement direction
- Useful because stock movement direction can matter more than exact return value

## 15. Training Controls

### 15.1 Epochs

An epoch means one full pass through the training data.

Recommended current command:

```powershell
venv\Scripts\python.exe hybrid_hydro_market_model.py --epochs 50 --batch-size 64
```

### 15.2 Batch Size

Batch size is the number of samples processed before updating model weights.

Current recommendation:

```text
batch_size = 64
```

Purpose:

- Larger batch size trains faster but may generalize less
- Smaller batch size trains slower but may learn more detailed patterns

### 15.3 Early Stopping

The model monitors validation loss:

```text
monitor = val_loss
patience = 8
```

Purpose:

- Stops training when validation performance stops improving
- Helps avoid overfitting

### 15.4 Learning Rate Reduction

The model reduces learning rate when validation loss plateaus:

```text
factor = 0.5
patience = 4
min_lr = 0.00001
```

Purpose:

- Allows the model to make smaller updates when learning slows down

## 16. How to Run

### 16.1 Install or Update Dependencies

The project dependencies are listed in:

```text
requirements.txt
```

Install or update the virtual environment with:

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Important direct dependencies:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
tensorflow
keras
scipy
ta
openpyxl
pillow
xgboost
```

Notes:

- `tensorflow` is required for the Regularized LSTM model.
- `scikit-learn` is required for Gradient Boosting, RobustScaler, and metrics.
- `xgboost` is required for the XGBoost regressor.
- `joblib` is used to save the Gradient Boosting and XGBoost models.
- `numpy`, `pandas`, and `matplotlib` are used throughout preprocessing, feature engineering, and chart generation.
- `seaborn` is required by the original price preprocessing script.

### 16.2 Regenerate Rainfall and Production Features

```powershell
python rainfall_production_preprocessing.py
```

### 16.3 Create Hydrology-Market Arrays Without Training

```powershell
python hybrid_hydro_market_model.py --dry-run
```

For a 20-trading-day target:

```powershell
python hybrid_hydro_market_model.py --target-horizon 20 --dry-run
```

### 16.4 Train Regularized LSTM, Gradient Boosting, and XGBoost

Use the virtual environment because TensorFlow is installed there:

```powershell
venv\Scripts\python.exe hybrid_hydro_market_model.py --epochs 50 --batch-size 64
```

For a 20-trading-day horizon:

```powershell
venv\Scripts\python.exe hybrid_hydro_market_model.py --target-horizon 20 --epochs 100 --batch-size 64
```

For a 60-trading-day horizon:

```powershell
venv\Scripts\python.exe hybrid_hydro_market_model.py --target-horizon 60 --epochs 100 --batch-size 64
```

### 16.5 Train One Company Only

```powershell
venv\Scripts\python.exe hybrid_hydro_market_model.py --companies CHCL --epochs 50 --batch-size 64
```

With a 20-trading-day horizon:

```powershell
venv\Scripts\python.exe hybrid_hydro_market_model.py --companies CHCL --target-horizon 20 --epochs 100 --batch-size 64
```

## 17. Interpretation Notes

The hydrological and production data are annual, while market data are daily. Therefore, the same annual rainfall and production values are attached to all daily market rows within that year.

To avoid relying only on static repeated annual values, the merged daily dataset now also includes seasonal features:

```text
monsoon_season
dry_season
month_of_year_sin
month_of_year_cos
generation_per_trading_day
rainfall_year_progress
```

This means:

- Hydrology and production features represent yearly context
- Seasonal features represent daily position within the hydrological year
- Market features represent daily price and volume behavior
- The model studies whether yearly operational/hydrological conditions improve prediction of daily market movement

Important limitation:

- Annual hydrology data may still be too low-frequency to explain very short-term market movement by itself
- It is more useful as contextual information combined with daily technical indicators
- Seasonal features improve the daily signal, but they are still engineered approximations rather than measured daily rainfall or daily production
- If the 5-day horizon is weak, use `--target-horizon 20` or `--target-horizon 60` because hydrological and production signals are more likely to affect monthly or quarterly movement

## 18. Experiment Findings and Training-Loss Review

The main research question is whether hydrological and production factors, combined with market features, help explain or predict hydropower stock movement.

The strongest results so far are company-specific rather than uniform across all companies.

### 18.1 Final Saved Output-Folder Comparison

The final comparison was read from the saved per-company result files:

```text
Output/HydroMarketStudy/hydro_market_model_results_BARUN.csv
Output/HydroMarketStudy/hydro_market_model_results_CHCL.csv
Output/HydroMarketStudy/hydro_market_model_results_RADHI.csv
```

A combined copy is also saved as:

```text
Output/HydroMarketStudy/hydro_market_model_results_ALL.csv
```

BARUN saved result:

```text
Regularized_LSTM      RMSE 0.154387 | MAE 0.129844 | R2  0.097077 | Direction 76.17%
Persistence_Baseline  RMSE 0.187461 | MAE 0.156011 | R2 -0.331231 | Direction 64.95%
GradientBoosting      RMSE 0.232189 | MAE 0.194288 | R2 -0.518488 | Direction 50.00%
XGBoost               RMSE 0.232970 | MAE 0.195532 | R2 -0.528717 | Direction 54.28%
```

Conclusion:

```text
BARUN is best explained by the Regularized LSTM. It beats the persistence baseline and both tree models on RMSE, MAE, R2, and directional accuracy.
```

CHCL saved result:

```text
XGBoost               RMSE 0.066956 | MAE 0.053379 | R2  0.298054 | Direction 79.02%
Regularized_LSTM      RMSE 0.076496 | MAE 0.064470 | R2  0.099828 | Direction 74.29%
Persistence_Baseline  RMSE 0.081365 | MAE 0.067555 | R2 -0.018409 | Direction 63.67%
GradientBoosting      RMSE 0.088807 | MAE 0.072349 | R2 -0.234882 | Direction 70.49%
```

Conclusion:

```text
CHCL is best explained by XGBoost. It has the best RMSE, MAE, R2, and directional accuracy among all saved CHCL models.
```

RADHI saved result:

```text
XGBoost               RMSE 0.184153 | MAE 0.130563 | R2 -0.313152 | Direction 66.10%
GradientBoosting      RMSE 0.184657 | MAE 0.130431 | R2 -0.320353 | Direction 67.80%
Persistence_Baseline  RMSE 0.199345 | MAE 0.152192 | R2 -0.392500 | Direction 48.09%
Regularized_LSTM      RMSE 0.205372 | MAE 0.133466 | R2 -0.477983 | Direction 43.83%
```

Conclusion:

```text
RADHI remains weak for return magnitude. XGBoost has the best RMSE and R2, while Gradient Boosting has the best directional accuracy. Both tree models beat the baseline directionally, but all RADHI R2 values remain negative.
```

Cross-company model comparison:

```text
BARUN best overall model = Regularized_LSTM
CHCL best overall model  = XGBoost
RADHI best magnitude     = XGBoost
RADHI best direction     = GradientBoosting
```

Overall finding:

```text
The best model is not the same for every company. BARUN benefits most from sequence learning, CHCL benefits most from XGBoost tabular learning, and RADHI shows only limited directional usefulness from tree models.
```

### 18.2 Training-Loss Chart Review

Charts reviewed:

```text
Output/HydroMarketStudy/BARUN_Regularized_LSTM_training_loss.png
Output/HydroMarketStudy/CHCL_Regularized_LSTM_training_loss.png
Output/HydroMarketStudy/RADHI_Regularized_LSTM_training_loss.png
```

BARUN chart:

```text
Training loss decreases steadily.
Validation loss falls until about epoch 6, then rises again.
```

Judgment:

```text
The chart is acceptable but not perfect. It shows useful learning early, followed by overfitting after the validation trough. EarlyStopping is appropriate. The BARUN LSTM result is usable because the saved test metrics beat all baselines.
```

CHCL chart:

```text
Training loss decreases while validation loss rises almost immediately.
```

Judgment:

```text
The chart shows clear LSTM overfitting. CHCL's best saved model is therefore not LSTM but XGBoost, which performs better without relying on a recurrent training curve.
```

RADHI chart:

```text
Validation loss improves briefly until the early epochs, then rises while training loss continues to fall.
```

Judgment:

```text
The RADHI LSTM is not reliable. The chart and metrics both show that LSTM does not generalize well for RADHI. XGBoost is slightly better for return magnitude, while Gradient Boosting is slightly better for direction.
```

Overall chart conclusion:

```text
The LSTM training curves are not perfect. They show overfitting pressure in all companies. EarlyStopping is essential and should remain enabled. The thesis should not claim that LSTM is universally best. It is best for BARUN, useful but not best for CHCL, and weak for RADHI.
```

### 18.3 Does This Support the Thesis?

The results partially support the thesis.

Supported:

- Hydrological and production-enhanced features are more useful at medium-term horizons than at 5 trading days.
- BARUN shows meaningful improvement with Regularized LSTM.
- CHCL shows the strongest saved result with XGBoost.
- RADHI shows modest directional improvement with tree models, especially Gradient Boosting.
- Seasonal hydrology features and configurable target horizons made the study more aligned with hydropower fundamentals.

Not fully supported:

- The relationship is not equally strong across all companies.
- RADHI does not show reliable return magnitude prediction.
- LSTM training curves show overfitting pressure.
- Negative R2 values for RADHI and some baseline/tree runs mean models can still fail to explain return magnitude.

Final thesis interpretation:

```text
The study supports the idea that hydrological and production factors can contribute to medium-term hydropower stock movement analysis when combined with market indicators. However, the relationship is company-specific, model-specific, and horizon-dependent. BARUN is best captured by LSTM sequence learning, CHCL is best captured by XGBoost tabular learning, and RADHI shows weaker magnitude prediction with only modest directional usefulness.
```

Recommended final framing:

```text
This thesis should conclude that hydrological and production factors are not reliable standalone predictors of short-term NEPSE hydropower movement, but they add useful contextual signal for selected companies over monthly-to-quarterly horizons.
```

## 19. Current Status

Completed:

- Rainfall and production preprocessing
- Feature engineering for hydrology and production
- Charts for rainfall and production
- Hydrology-market merged features
- Daily seasonal hydrology features
- TensorFlow-ready sequence arrays
- Regularized LSTM, Gradient Boosting, and XGBoost training script
- Configurable target horizon testing
- Company-wise model comparison
- Training-loss chart review
- Updated `requirements.txt` with direct dependencies, including `numpy`, `seaborn`, and `xgboost`
- Final combined model comparison saved to `Output/HydroMarketStudy/hydro_market_model_results_ALL.csv`
- Relation summary between hydrology/production features and configurable-horizon market return

Next recommended steps:

- Preserve best results in a separate final results table because `hydro_market_model_results.csv` is overwritten by each run
- Repeat best configurations 3 to 5 times to check neural-model stability
- Analyze `hydro_market_relation_summary.csv`
- Compare model performance with and without hydrology/production features
- Write final thesis discussion using company-specific and horizon-specific conclusions
