# NEPSE Hydropower Stock Analysis Based on Hydrological and Operational Factors Using Machine Learning

**Major Project — README & Viva / Defense Preparation Guide**

This README is written specifically to help you (a) understand the whole project end-to-end and
(b) walk into the viva/defense able to answer almost any question an examiner can ask about the
topic, the data, the methodology, the models, the results, and the limitations.

It is organised in two parts:

- **Part A — Project Reference:** what the project is, how it is built, how to run it.
- **Part B — Anticipated Viva Questions with Model Answers:** ~70 questions grouped by theme,
  each with a ready answer you can study and adapt.

> Tip for the report: this project maps cleanly onto the Amity `PROJECT FORMAT.docx` chapter
> structure. A chapter-by-chapter mapping is given in [Section 9](#9-mapping-to-the-amity-report-format).

---

## Part A — Project Reference

## 1. Project at a Glance

| Item | Detail |
|------|--------|
| **Title** | NEPSE Hydropower Stock Analysis Based on Hydrological and Operational Factors Using Machine Learning |
| **Domain** | Financial time-series forecasting + hydrology + machine learning |
| **Market** | NEPSE (Nepal Stock Exchange), hydropower sector |
| **Companies studied** | **BARUN** (Barun Hydropower), **CHCL** (Chilime Hydropower), **RADHI** (Radhi Bidyut) |
| **Core question** | Do **hydrological** (rainfall) and **operational** (energy generation) factors help explain or predict short-to-medium-term hydropower stock movement when combined with market technical indicators? |
| **Prediction target** | Future return over a configurable horizon (default 5 trading days; also 20 and 60) |
| **Models compared** | Persistence baseline, Regularized **LSTM**, **Gradient Boosting**, **XGBoost** |
| **Language / stack** | Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, XGBoost, `ta`, matplotlib/seaborn |

### Data coverage (actual, from the repo)

| Company | Daily market rows | Date range |
|---------|------------------:|------------|
| BARUN | 2,679 | 2015-10-14 → 2026-04-23 |
| CHCL  | 4,954 | 2007-01-02 → 2026-04-23 |
| RADHI | 2,045 | 2018-03-26 → 2026-04-23 |

- **Annual rainfall & production:** 2018–2025 for all three companies (24 rows total).
- **Fundamentals:** quarterly balance-sheet / income-statement / key-stats data per company.

---

## 2. Repository Structure

```
.
├── Datasets/                         # Raw inputs
│   ├── rainfall&production.csv       # Annual rainfall (mm) + energy generation (GWh)
│   ├── Fundamentals/                 # BARUN.csv, CHCL.csv, RADHI.csv (quarterly financials)
│   └── Price Volume/                 # OHLCV: per company OPEN/HIGH/LOW/CLOSE/ADJ CLOSE/VOLUME.csv
│
├── price_preprocessing.py            # Cleans OHLCV, engineers ~80 market features, makes plots
├── rainfall_production_preprocessing.py  # Engineers annual hydrology + production features
├── fundamentals_preprocessing.py     # Parses quarterly financial statements into ratios
├── hybrid_hydro_market_model.py      # MAIN: merges everything, trains & compares 3 models
│
├── Features/                         # Engineered feature tables
│   ├── Production/  Rainfall/  HydroMarket/  Fundamentals/  Price Volume/
│
├── Output/                           # All plots, arrays, and results
│   ├── {COMPANY}_features.csv        # Daily market feature tables (87 columns)
│   ├── *.png                         # Thesis-quality EDA charts
│   └── HydroMarketStudy/             # Trained models, prediction charts, result CSVs
│
├── requirements.txt
├── PROJECT_WORK_DOCUMENTATION.md     # Deep technical log (formulas + findings)
└── README.md                         # ← this file
```

---

## 3. Data Sources and Meaning

1. **Market data (OHLCV)** — daily Open, High, Low, Close, Adjusted Close, Volume per company,
   stored as one CSV per field under `Datasets/Price Volume/{COMPANY}/`.
2. **Hydrological data** — `Annual Rainfall (mm)` measured at a rainfall station near each
   plant's catchment (CHCL: Dhunche, Rasuwa; RADHI: Khudi Bazar, Lamjung; BARUN: Chainpur,
   Sankhuwasabha).
3. **Operational data** — `Energy Generation (GWh)` per company per year (the plant's actual
   electricity output).
4. **Fundamental data** — quarterly balance sheet (BS), income statement (IS), and key stats (KS),
   used to derive ratios such as EPS, ROE, ROA, current ratio, debt-to-equity, margins, QoQ growth.

> **Why these three companies?** They are listed NEPSE hydropower firms with reasonably long,
> usable price histories and identifiable nearby rainfall stations, so rainfall, generation, and
> market data can be aligned.

---

## 4. The Four Processing Scripts

### 4.1 `price_preprocessing.py`
Loads the six OHLCV files per company, runs a **data-quality report** (gaps, duplicates, OHLC
violations, zero volume), **cleans** (business-day reindex, forward-fill ≤2 days, linear
interpolate ≤5 days, OHLC consistency fix), then **engineers ~80 features**:

- Returns (1/2/3/5/10/20-day), log return, overnight gap, intraday range, close-open return
- Rolling stats (MA, std, min, max, price-relative-to-MA, volume MA/std, return skew/kurtosis) over windows 5/10/20/50
- Technical indicators via `ta`: EMA, MACD (+signal/diff), ADX, RSI (7/14/21), Stochastic, CCI, Williams %R, Bollinger Bands, ATR, OBV, CMF, VWAP proxy
- Volume features, calendar/seasonality features (incl. Nepali fiscal year), price-structure ratios, lag features

It creates targets (5-day return, log return, direction), winsorises return outliers (IQR × 3),
does a **temporal 70/15/15 split**, **RobustScaler** (fit on train only), builds **60-day LSTM
sequences**, and saves `Output/{COMPANY}_features.csv`, `*_arrays.npz`, and 10 thesis plots.

### 4.2 `rainfall_production_preprocessing.py`
Fixes the malformed CSV header (the comma inside `Rainfall Station (District, Elevation)`),
then engineers **annual hydrology and production features** per company: lags (1y/2y), YoY change
(absolute & %), 3-year rolling mean/std, expanding mean, deviation from company mean, z-score,
cumulative totals, **generation per mm of rainfall** (efficiency), and rainfall/generation
indices. Saves combined + per-company feature files and five `RP_*` charts.

### 4.3 `fundamentals_preprocessing.py`
A `FundamentalsPreprocessor` class that melts wide quarterly statements into tidy `(Company,
Period)` rows, parses Nepali fiscal periods (e.g. `2024/25 (2081/82) Q3`), and derives ratios
(current ratio, debt-to-equity, gross/net/operating margin, effective tax rate, ROA/ROE, QoQ &
YoY growth). Output feeds `Features/Fundamentals/`.

### 4.4 `hybrid_hydro_market_model.py` (the main study)
The heart of the project. It:

1. Loads each company's daily market features and the annual rainfall + production features.
2. **Merges** annual hydrology/production into daily rows by `Company + Year` (the annual value
   repeats across that year's trading days).
3. Adds **daily seasonal hydro features** (monsoon flag, dry-season flag, cyclic month sin/cos,
   generation-per-trading-day, year-progress) so the model isn't fed a flat annual constant.
4. **Imputes** hydro lag/rolling NaNs with an expanding historical mean (never zero-fill).
5. Builds the regression target `target_return_{h}d`, does the temporal split, RobustScales
   features and target, and builds 60-day sequences.
6. Trains and compares: **Persistence baseline**, **Regularized LSTM**, **Gradient Boosting**,
   **XGBoost**, with class-imbalance-aware sample weighting for the tree models.
7. Saves models, prediction charts, training-loss charts, a feature↔target correlation summary,
   and per-company + combined results CSVs in `Output/HydroMarketStudy/`.

---

## 5. The Models (Summary)

| Model | Type | Input shape | Why it's here |
|-------|------|-------------|---------------|
| **Persistence baseline** | Naive | last same-horizon return | Sanity floor — any useful model must beat it |
| **Regularized LSTM** | Deep sequence | (60 days × features) | Learns temporal market memory |
| **Gradient Boosting** | Tree ensemble (sklearn) | flat daily row | Strong, interpretable tabular baseline |
| **XGBoost** | Tree ensemble (regularized) | flat daily row | Stronger tabular model with L1/L2 + subsampling |

**Regularized LSTM architecture:** `Input(60, n) → LSTM(48, dropout 0.20, recurrent 0.10) →
LayerNorm → Dropout(0.30) → LSTM(24) → GlobalAveragePooling1D → Dense(16, relu) → Dropout(0.15)
→ Dense(1)`. Loss = **Huber**, optimizer = Adam, with **EarlyStopping** (patience 8) and
**ReduceLROnPlateau** (factor 0.5).

**Evaluation metrics:** RMSE, MAE, R², and **directional accuracy** (did we get the up/down sign
right). Targets are inverse-transformed to the real return scale before metrics are computed.

---

## 6. Headline Results (from saved result CSVs)

| Company | Best model | RMSE | R² | Directional accuracy |
|---------|-----------|-----:|---:|---------------------:|
| **BARUN** | Regularized LSTM | 0.154 | **+0.097** | **76.2%** |
| **CHCL**  | XGBoost          | 0.067 | **+0.298** | **79.0%** |
| **RADHI** | XGBoost (magnitude) / GB (direction) | 0.184 | −0.313 | 66–68% |

**Key takeaway:** No single model wins for every company. BARUN is best captured by sequence
learning (LSTM), CHCL by tabular XGBoost, and RADHI shows only modest directional usefulness with
weak return-magnitude prediction (negative R²). All best models beat the persistence baseline on
directional accuracy.

**Overall conclusion:** Hydrological and operational factors are **not reliable stand-alone
predictors** of short-term NEPSE hydropower movement, but they add **useful contextual signal**
for selected companies over **monthly-to-quarterly horizons** — i.e. the relationship is
*company-specific, model-specific, and horizon-dependent.*

---

## 7. How to Run

```powershell
# 1. Create / activate venv and install dependencies
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Build market features (OHLCV → features + plots)
venv\Scripts\python.exe price_preprocessing.py

# 3. Build annual rainfall + production features
venv\Scripts\python.exe rainfall_production_preprocessing.py

# 4. Run the hybrid study (default 5-day horizon)
venv\Scripts\python.exe hybrid_hydro_market_model.py --epochs 50 --batch-size 64

# Longer, more hydrology-relevant horizons
venv\Scripts\python.exe hybrid_hydro_market_model.py --target-horizon 20 --epochs 100 --batch-size 64
venv\Scripts\python.exe hybrid_hydro_market_model.py --target-horizon 60 --epochs 100 --batch-size 64

# Single company, or just build arrays without training
venv\Scripts\python.exe hybrid_hydro_market_model.py --companies CHCL --epochs 50
venv\Scripts\python.exe hybrid_hydro_market_model.py --dry-run
```

---

## 8. Requirements

`numpy, pandas, matplotlib, seaborn, scikit-learn, joblib, tensorflow, keras, scipy, ta, openpyxl,
pillow, xgboost` (exact pins in `requirements.txt`). TensorFlow is needed for the LSTM, scikit-learn
for Gradient Boosting + scaling + metrics, and XGBoost for the XGBoost regressor.

---

## 9. Mapping to the Amity Report Format

`PROJECT FORMAT.docx` requires these chapters. Here is where your material comes from:

| Report Section | What to write / where it comes from |
|----------------|-------------------------------------|
| **Abstract** (500–1000 words) | Summarise aim, data, the 4 models, headline results, and the company-specific conclusion. |
| **Ch.1 Introduction** | NEPSE + hydropower sector context; why rainfall→generation→cash-flow→stock price is plausible; justification for the topic. |
| **Ch.2 Literature Review** | ML for stock prediction (LSTM/GRU, gradient boosting, XGBoost); hydropower & rainfall studies; technical-indicator forecasting. |
| **Ch.3 Objectives & Methodology** | Objectives (Q1 below); research design (quantitative, secondary data); data type/sources/collection; sample = 3 companies; data-analysis tools = Python ML stack. |
| **Ch.4 Data Analysis & Results** | EDA charts (`Output/*.png`), feature↔target correlations, the model comparison tables, prediction & training-loss charts. |
| **Ch.5 Findings & Conclusion** | The company-specific findings in [Section 6](#6-headline-results-from-saved-result-csvs) and Part B. |
| **Ch.6 Recommendations & Limitations** | See viva Q60–Q70. |
| **Bibliography / Appendix** | APA references + feature formula appendix (in `PROJECT_WORK_DOCUMENTATION.md`). |

---

# Part B — Anticipated Viva Questions with Model Answers

> Study these. Examiners typically probe: *why this topic*, *why these methods*, *what the data
> really is*, *whether the model is valid*, and *what the results actually mean (and don't mean)*.

## A. Topic, Motivation, Objectives

**Q1. What are the objectives of your study?**
(1) To engineer a combined dataset of market, hydrological, operational, and seasonal features for
NEPSE hydropower stocks. (2) To investigate whether rainfall and energy-generation factors are
correlated with / help predict future stock returns. (3) To build and compare deep-learning
(LSTM) and machine-learning (Gradient Boosting, XGBoost) models against a naive baseline across
multiple prediction horizons (5, 20, 60 trading days). (4) To identify which model and horizon
work best for each company and to interpret the practical and financial meaning.

**Q2. Why did you choose this topic?**
Hydropower dominates Nepal's electricity sector and a meaningful part of NEPSE. A hydropower
company's revenue depends physically on water availability (rainfall → river flow → generation →
energy sales → profit). It is therefore intuitive to test whether hydrological and operational
data carry predictive signal for these stocks — a link rarely studied for NEPSE specifically.

**Q3. What is the central research question / hypothesis?**
Hypothesis: hydrological and operational factors, combined with market technical indicators,
improve prediction of hydropower stock movement. The study tests this and finds it is only
*partially* supported — the effect is company- and horizon-specific, not universal.

**Q4. Why hydropower specifically and not banking or other sectors?**
Because hydropower has a clear, physical external driver (rainfall/generation) that other sectors
lack. This makes it the ideal sector to test whether *non-market* environmental/operational data
adds value beyond standard technical indicators.

**Q5. Why these three companies — BARUN, CHCL, RADHI?**
They are listed NEPSE hydropower firms with (a) sufficiently long, clean daily price histories,
and (b) an identifiable rainfall station near the plant's catchment, so rainfall, generation, and
market data can be geographically and temporally aligned. CHCL (Chilime) is large and long-listed;
BARUN and RADHI are smaller/newer, giving useful contrast.

## B. Data

**Q6. What data did you use and what are the sources?**
Daily OHLCV market data per company (NEPSE), annual rainfall (mm) from nearby DHM-type rainfall
stations, annual energy generation (GWh) per plant, and quarterly fundamental financial statements.
All are **secondary data**.

**Q7. What is the time span of your data?**
Market: CHCL 2007→2026, BARUN 2015→2026, RADHI 2018→2026 (daily). Rainfall and generation: annual
2018–2025. The overlap window after merging is therefore effectively 2018 onward.

**Q8. What is your sample size?**
Three companies (the analytical units). In rows: ~2,000–5,000 daily observations per company before
merging; 8 annual hydrology/production records per company.

**Q9. Your market data is daily but rainfall/generation is annual. Isn't that a mismatch?**
Yes — this is the project's central data-frequency challenge. The same annual value repeats across
all trading days in that year. I handle it two ways: (1) I keep the annual value as *yearly context*,
and (2) I add **daily seasonal features** (monsoon/dry flags, cyclic month encoding, year-progress,
generation-per-trading-day) so the model still receives time-varying hydrological signal. I am also
explicit in the limitations that annual data is low-frequency for short horizons.

**Q10. How did you handle missing values and bad data in the market data?**
The quality report flags gaps, duplicates, OHLC violations, and zero-volume days. Cleaning:
de-duplicate, reindex to business days, set negatives to NaN, swap inverted high/low, forward-fill
short gaps (≤2 days), linear-interpolate medium gaps (≤5 days), then drop any remaining missing
close.

**Q11. How did you handle missing hydrology lag features (e.g. the first years have no lag)?**
`impute_hydro_lags()` fills NaNs in lag/rolling columns with the **expanding historical mean** (mean
of values available *before* that row), never zero — because zero would falsely signal "no
production/rainfall." A binary `hydro_lag_imputed` flag marks imputed rows.

**Q12. Is the rainfall data real measured data?**
The earlier years are based on station records; some later years appear to be reasonable
estimates/round figures. I disclose this in the limitations — measured daily rainfall would be
stronger than annual figures.

## C. Feature Engineering

**Q13. How many features do you have and of what kinds?**
~87 columns in the market feature table. Groups: (1) **market/technical** — returns, moving
averages, volatility, MACD, RSI, Bollinger Bands, volume features; (2) **hydrology** — rainfall
lags, YoY change, rolling stats, z-scores, indices; (3) **production/operational** — generation
lags, YoY, efficiency (generation per mm rainfall); (4) **seasonal** — monsoon/dry flags, cyclic
month, year-progress.

**Q14. Explain a few key technical indicators you used.**
*RSI* (momentum oscillator, overbought >70 / oversold <30); *MACD* (EMA12−EMA26, momentum/trend
shift); *Bollinger Bands* (MA20 ± 2σ, volatility/price position); *moving averages* (trend);
*volume surge* (today's volume ÷ 20-day average, unusual activity).

**Q15. What is "generation per mm of rainfall" and why compute it?**
`generation_per_mm_rainfall = Energy Generation (GWh) ÷ Annual Rainfall (mm)`. It's a rough
**operational efficiency** indicator — how much electricity the plant produced per unit of rainfall
that year — letting the model compare efficient vs inefficient years.

**Q16. Why cyclic (sin/cos) encoding for the month?**
So December (12) and January (1) are mathematically *close* rather than maximally far apart. Raw
month numbers imply a false 11-unit jump from Dec→Jan; sin/cos puts months on a circle so the model
learns true annual seasonality.

**Q17. Why monsoon and dry-season flags?**
Nepal's rainfall is concentrated in the monsoon (Jun–Sep). These flags let the model distinguish
wet-season trading periods (when generation pressure differs) from dry-season periods, adding daily
hydrological context the annual figure cannot.

**Q18. Did you check for multicollinearity among features?**
Yes — feature correlation heatmaps are generated (`*_09_feature_heatmap.png`). Many technical
indicators are correlated by construction; tree models (GB/XGBoost) are robust to this, and the
LSTM's regularisation/dropout mitigates it. RobustScaler also standardises scales.

## D. Methodology & Validity

**Q19. Why did you use a temporal (time-based) train/val/test split instead of random?**
Because this is time-series forecasting. Random shuffling would leak *future* information into the
training set, inflating accuracy unrealistically. I use the first 70% for training, next 15% for
validation, last 15% for testing — the model trains on the past and is tested on unseen newer data,
which mirrors real deployment.

**Q20. What is the target variable?**
`target_return_{h}d = (close[t+h] − close[t]) / close[t]` — the future return over h trading days
(regression). A direction target (1 if price rises, else 0) is also derived. Default h=5; also 20
and 60.

**Q21. Why test multiple horizons (5, 20, 60 days)?**
Rainfall and generation are annual/seasonal signals. Their effect is unlikely to surface in 5 days
but may matter over a trading month (≈20 days) or quarter (≈60 days). Testing multiple horizons
matches the model to the timescale of the hydrological driver.

**Q22. Why a 60-day sequence length for the LSTM?**
It gives the LSTM ~3 trading months of recent history per prediction, enough to learn short-term
momentum and trend memory without making sequences so long that the small dataset can't support
them.

**Q23. Why RobustScaler instead of StandardScaler or MinMaxScaler?**
RobustScaler centres on the median and scales by the IQR, so it is **insensitive to outliers** —
important because stock returns and volume have extreme spikes that would distort mean/variance-based
scalers. The scaler is fit on **training data only**, then applied to val/test (no leakage).

**Q24. Why scale the target as well as the features for the LSTM?**
Scaling the target helps the neural network converge. Crucially, I **inverse-transform** predictions
back to the real return scale *before* computing metrics, so RMSE/MAE/R² are meaningful in actual
return units.

**Q25. How do you prevent data leakage?**
(a) Temporal split, no shuffling; (b) scalers fit on training only; (c) targets are future-shifted
and rows with unknown future are dropped; (d) lag/rolling features only use past values; (e)
expanding-mean imputation uses only prior rows.

**Q26. What is the persistence baseline and why include it?**
It predicts the future h-day return using the most recent trailing h-day return ("tomorrow looks
like yesterday"). It's the naive floor — a model is only genuinely useful if it beats this baseline.

## E. The Models

**Q27. Why did you choose LSTM?**
LSTM is designed for sequential data: its input/forget/output gates let it retain or discard
information over a sequence, capturing temporal dependencies in price movement that a plain
feed-forward model cannot.

**Q28. Explain your LSTM architecture.**
`Input(60×features) → LSTM(48, return_sequences) → LayerNormalization → Dropout(0.30) → LSTM(24,
return_sequences) → GlobalAveragePooling1D → Dense(16, relu) → Dropout(0.15) → Dense(1)`. The
stacked LSTMs learn sequence patterns; LayerNorm stabilises training; dropout/recurrent-dropout
fight overfitting; global average pooling summarises the sequence; the final Dense outputs the
predicted return.

**Q29. Why is it called the "Regularized" LSTM?**
Because it is deliberately small and heavily regularised — dropout (0.20/0.30/0.15), recurrent
dropout (0.10), LayerNormalization, Huber loss, EarlyStopping, and ReduceLROnPlateau — all to combat
the overfitting seen with the earlier larger CNN-LSTM/GRU design on this small dataset.

**Q30. Why did you switch from CNN-LSTM / GRU to this LSTM?**
The initial CNN-LSTM was too flexible and overfit (training loss fell while validation loss rose).
A simple GRU was tried but barely beat the persistence baseline. The regularised LSTM is a
middle ground: smaller than CNN-LSTM, with stronger explicit memory than the GRU.

**Q31. Why Huber loss instead of MSE for the LSTM?**
Huber behaves like MSE for small errors but like MAE for large ones, so it is **less sensitive to
outlier returns** — more stable for noisy financial targets than pure MSE.

**Q32. What is Gradient Boosting and how does it work?**
An ensemble that builds many shallow decision trees **sequentially**, where each new tree corrects
the residual errors of the previous ensemble. The summed trees give the prediction. It handles
nonlinear relationships well on tabular data and needs no sequences.

**Q33. What is XGBoost and how does it differ from sklearn Gradient Boosting?**
XGBoost is an optimised gradient-boosting library with **L1/L2 regularisation** (`reg_alpha`,
`reg_lambda`), **column and row subsampling**, second-order gradient optimisation, and a fast
histogram tree method. It is typically stronger and less prone to overfitting on tabular data than
vanilla Gradient Boosting.

**Q34. Why compare a deep-learning model with tree models?**
To test whether the hydrology-market relationship is better captured by **sequence learning**
(LSTM) or **flat tabular learning** (GB/XGBoost). If a tree model wins, it suggests the signal lives
in the current day's engineered features rather than in 60-day temporal patterns.

**Q35. The LSTM uses 60-day sequences but the tree models use flat rows — is that a fair comparison?**
They are different but complementary representations of the same information: the LSTM sees the
recent *sequence*, the tree models see the current engineered row (which already contains lagged and
rolling features). I report both so the reader sees which representation works best per company.

**Q36. What is the class-imbalance sample weighting in the tree models?**
Up-days and down-days are not equally frequent. I weight the minority class higher (inversely
proportional to its frequency) so the model doesn't just predict the majority direction.

**Q37. What hyperparameters did you use for the tree models?**
GB: 300 trees, learning rate 0.03, max depth 2, min_samples_leaf 8, subsample 0.80. XGBoost: 500
trees, lr 0.03, max depth 2, min_child_weight 8, subsample/colsample 0.80, reg_lambda 5.0,
reg_alpha 0.10. Shallow trees + low learning rate + subsampling = strong regularisation for a small
dataset.

## F. Metrics & Evaluation

**Q38. Which evaluation metrics did you use and why?**
**RMSE** (error in return units, penalises big misses), **MAE** (average absolute error, easy to
interpret), **R²** (variance explained), and **directional accuracy** (fraction of correct up/down
calls). For trading, *direction* often matters more than exact magnitude, so directional accuracy is
emphasised.

**Q39. Several R² values are negative. What does that mean and is your project a failure?**
A negative R² means the model predicts return *magnitude* worse than simply predicting the test-set
mean. It is honestly reported, not hidden. It does **not** make the project a failure: (a) predicting
exact returns is extremely hard and negative R² is common in real return forecasting; (b) the same
models still achieve **66–79% directional accuracy**, beating the baseline; (c) the study's value is
the comparative finding about *which* factors/models/horizons help, not a claim of perfect
prediction.

**Q40. What is directional accuracy and why highlight it?**
`directional_accuracy = (count where sign(predicted) == sign(actual)) / total`. It measures whether
the model gets the *direction* of movement right. CHCL reaches 79% and BARUN 76%, which is the
practically useful result for a trader who cares about up vs down.

**Q41. How do you know a result is good and not luck?**
Each model is compared against the persistence baseline on the *same* test set; the best models beat
it on directional accuracy (and on RMSE/MAE for BARUN/CHCL). The documentation also recommends
repeating neural runs 3–5 times to check stability — acknowledged as future work.

## G. Results & Interpretation

**Q42. What are your main findings?**
(1) BARUN is best predicted by the Regularized LSTM (R² +0.097, 76% direction). (2) CHCL is best
predicted by XGBoost (R² +0.298, 79% direction) — the strongest result. (3) RADHI is weak on
magnitude (negative R²) but tree models give 66–68% directional accuracy. (4) No single model wins
everywhere.

**Q43. Why does CHCL perform best?**
CHCL (Chilime) is the largest, longest-listed, and most-traded of the three, so it has the most
data and the cleanest price signal. More data + steadier fundamentals make its patterns easier for
XGBoost to learn.

**Q44. Why does RADHI perform worst?**
RADHI is smaller, with a shorter history (from 2018) and noisier, less liquid price action, so the
signal-to-noise ratio is lower and return magnitude is hard to predict — though direction is still
better than chance.

**Q45. Why does the best model differ across companies?**
Because each stock has different dynamics: BARUN's movement depends on patterns spread across the
60-day sequence (favouring LSTM), while CHCL's signal lives more in the current engineered feature
row (favouring XGBoost). This is itself a key finding: there is no universal best model.

**Q46. Did the training curves look healthy?**
Not perfectly. BARUN's validation loss improves until ~epoch 6 then rises (mild overfitting, handled
by EarlyStopping). CHCL's and RADHI's LSTM validation loss rise early, showing LSTM overfitting —
which is exactly why their best models are tree-based, not the LSTM.

**Q47. So did hydrology/production data actually help?**
Partially. The feature↔target correlation summary and the horizon experiments show hydrological and
operational features add **useful contextual signal for some companies at monthly/quarterly
horizons**, but they are **not reliable stand-alone short-term predictors**. The honest conclusion
is that they are a helpful *supplement* to market indicators, not a replacement.

**Q48. Did you compare performance with vs without hydrology features?**
That explicit ablation is listed as a recommended next step in the documentation. The current
evidence is indirect (feature-target correlations and horizon sensitivity); a clean with/without
ablation would strengthen the causal claim and is acknowledged future work.

**Q49. What does the feature↔target correlation summary tell you?**
It ranks each hydrology/production feature by its Pearson correlation with the future return
(`hydro_market_relation_summary_*.csv` and the correlation bar charts). Correlations are generally
weak, consistent with the finding that these factors are contextual rather than dominant.

## H. Conceptual / Theory Checks

**Q50. Why is stock-return prediction so hard (efficient market hypothesis)?**
The semi-strong EMH says prices already reflect public information, so future returns are close to a
random walk. That's why beating a persistence baseline at all, and reaching 70%+ directional
accuracy, is non-trivial — and why negative R² on magnitude is expected.

**Q51. What is overfitting and how did you control it?**
Overfitting is when a model memorises training noise and fails to generalise (training loss ↓ while
validation loss ↑). Controls: small/regularised architectures, dropout, LayerNorm, EarlyStopping,
ReduceLROnPlateau, shallow trees + subsampling for GB/XGBoost, and RobustScaling.

**Q52. What is the difference between regression and classification here?**
Regression predicts the *value* of the future return; the direction target is a *classification*
(up/down). I train regression models and derive direction from the sign of the predicted return,
then report both magnitude (RMSE/MAE/R²) and direction (accuracy).

**Q53. What is the bias–variance tradeoff in your models?**
The earlier CNN-LSTM had low bias but high variance (overfit). I deliberately moved toward
higher-bias, lower-variance designs (small regularised LSTM, shallow boosted trees) because the
dataset is small and noisy, where controlling variance matters more.

**Q54. Why not use a simple linear/ARIMA model?**
Linear/ARIMA models assume linear relationships and struggle with the many nonlinear technical and
hydrological interactions here. The persistence baseline already captures the naive linear
intuition, and I test whether nonlinear ML can beat it.

**Q55. Could you have used Transformers or other architectures?**
Yes — Transformers, Temporal Convolutional Networks, or attention models are possible. They were not
used because the dataset is small (they are data-hungry and overfit easily here) and because the
project's goal is a fair, interpretable comparison of established models. They are noted as future
work.

## I. Limitations, Recommendations, Misc

**Q56. What preprocessing decisions could bias your results?**
Forward-fill/interpolation of price gaps, winsorising return outliers, and especially repeating
annual hydrology values across daily rows. Each is justified but introduces assumptions; I disclose
them so results are interpreted cautiously.

**Q57. Why is the annual-to-daily repetition a problem and how serious is it?**
It means ~250 daily rows per year share identical hydrology values, which weakens the daily
predictive signal and can let the model "lean" on a per-year constant. It's the project's biggest
data limitation; the seasonal features partly compensate, and longer horizons reduce its impact.

**Q58. What are the main limitations of the study?** *(report-ready list)*
(1) Only 3 companies. (2) Annual hydrology vs daily market frequency mismatch. (3) Some rainfall/
generation values are estimates. (4) Short overlapping window (2018+). (5) Negative R² for RADHI. (6)
LSTM overfitting tendency. (7) No live trading/transaction-cost test. (8) Fundamentals not yet fully
integrated into the predictive model. (9) Single-run results (no repeated-seed averaging yet).

**Q59. What are your recommendations / future work?**
(1) Add measured *monthly/daily* rainfall and river-flow data. (2) Run a clean with/without-hydrology
ablation. (3) Integrate fundamentals (EPS/PE/ROE) into the predictive model. (4) Expand to more
hydropower companies. (5) Repeat neural runs with multiple seeds for stability. (6) Try
attention/Transformer models and SHAP explainability. (7) Test on the 20- and 60-day horizons
systematically. (8) Add a backtest with transaction costs.

**Q60. How can investors / the sector actually use this?**
As a **decision-support supplement**: directional models (70%+ for CHCL/BARUN) can flag likely
up/down bias over weeks-to-months, and hydrology context (wet vs dry year, generation efficiency)
can inform medium-term views — but it should not be used as a stand-alone automatic trading signal.

**Q61. Is this financial advice / can it guarantee profit?**
No. It is an academic study. Markets are noisy and partly efficient; the models are imperfect
(negative magnitude R² for some companies) and untested in live trading with costs. It informs
analysis, it does not guarantee returns.

**Q62. Why Python and these libraries?**
Python is the standard for ML/data science with mature, well-documented libraries: pandas/NumPy
(data), scikit-learn (ML + scaling + metrics), TensorFlow/Keras (deep learning), XGBoost (boosting),
`ta` (technical indicators), matplotlib/seaborn (visualisation).

**Q63. What is your research design and data-analysis tool? (report Ch.3 wording)**
Quantitative, secondary-data, comparative experimental design. Data-analysis tool: a custom Python
machine-learning pipeline (pandas, scikit-learn, TensorFlow/Keras, XGBoost) with temporal validation
and standard regression/classification metrics.

**Q64. How is reproducibility ensured?**
Fixed random seed (42) set across `random`, NumPy, and TensorFlow; deterministic temporal split;
saved scalers, arrays, models, and result CSVs; pinned dependency versions in `requirements.txt`.

**Q65. What does the `.npz` / `.joblib` / `.keras` files contain?**
`.npz` = saved NumPy arrays (scaled features, sequences, targets, baseline). `.joblib` = serialised
scikit-learn / XGBoost models. `.keras` = saved trained LSTM model. These let results be reloaded
without retraining.

**Q66. Why is generation roughly flat year-to-year for CHCL but rainfall varies?**
Run-of-river plants have a generation ceiling (installed capacity and plant efficiency), so output
saturates even when rainfall rises. This is itself an insight: above a threshold, *more rain ≠ more
generation*, weakening the rainfall→generation→price chain.

**Q67. How did you validate the link between rainfall and generation?**
Via the `RP_03_rainfall_vs_production` scatter, the dual-axis charts, and the
generation-per-mm-rainfall efficiency feature. The relationship exists but is non-linear and
capacity-capped, as above.

**Q68. If you had more time, what is the single most impactful improvement?**
Obtaining **measured monthly (or daily) rainfall and river-discharge data**. That single change
would remove the annual-to-daily repetition problem and most directly strengthen the core
hypothesis.

**Q69. What did you personally learn / contribute?**
An end-to-end, leakage-safe ML pipeline integrating heterogeneous data (market + hydrology +
operations + fundamentals) for NEPSE hydropower; a fair multi-model, multi-horizon comparison; and
an honest, nuanced, company-specific conclusion rather than an over-claimed "model predicts the
market."

**Q70. In one sentence, what is your conclusion?**
Hydrological and operational factors are not reliable stand-alone predictors of short-term NEPSE
hydropower stock movement, but combined with market indicators they provide useful contextual signal
for selected companies over monthly-to-quarterly horizons — the relationship is company-, model-,
and horizon-specific.

---

## Appendix — Quick Answer Cheatsheet

| If asked… | One-line answer |
|-----------|-----------------|
| Best overall result | CHCL with XGBoost: R² +0.30, 79% directional accuracy |
| Why temporal split | Prevent future-data leakage in time series |
| Why RobustScaler | Outlier-resistant (median/IQR), fit on train only |
| Why Huber loss | Robust to outlier returns vs MSE |
| Why negative R² is OK | Magnitude is hard; direction (66–79%) still beats baseline |
| Biggest limitation | Annual hydrology vs daily market frequency mismatch |
| Main conclusion | Useful contextual signal, not a stand-alone predictor; company/horizon specific |

*For full feature formulas, model rationale, and the detailed training-loss review, see
`PROJECT_WORK_DOCUMENTATION.md`.*
