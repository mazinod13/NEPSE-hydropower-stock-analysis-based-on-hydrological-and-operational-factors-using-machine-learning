# Report Writing Template

Use this as the writing skeleton for the format in `PROJECT FORMAT.docx`.

---

## TITLE PAGE

**Title:** NEPSE Hydropower Stock Analysis Based on Hydrological and Operational Factors Using Machine Learning

**Submitted by:**  
Name: __________  
Enrollment No.: __________

**Guide Details:**  
Name: __________  
Designation: __________

---

## ABSTRACT

Write 500-1000 words covering:

- project background
- problem statement
- datasets used
- methodology
- models used
- major findings
- conclusion

**Keywords:** NEPSE, hydropower stocks, rainfall, production, machine learning, feature engineering

---

## CHAPTER 1: INTRODUCTION TO THE TOPIC

Include:

- background of NEPSE hydropower sector
- why hydropower stocks were selected
- importance of rainfall, generation, and operational variables
- need for machine learning in stock analysis
- objective of combining hydrological and market factors

Possible figure:

- `Output/ALL_00_pipeline_diagram.png`

---

## CHAPTER 2: REVIEW OF LITERATURE

Include:

- prior studies on stock prediction using machine learning
- literature on hydropower generation and rainfall relationship
- literature on financial indicators and price movement
- research gap for Nepal hydropower market

No major dataset screenshots needed here unless used as a conceptual reference.

---

## CHAPTER 3: RESEARCH OBJECTIVES AND METHODOLOGY

### Research Objectives

- To analyze the relationship between rainfall, power generation, and hydropower stock behavior.
- To engineer meaningful features from hydrological, market, and fundamental datasets.
- To evaluate machine learning models for hydropower stock analysis and prediction.

### Research Problem

Write 1-2 paragraphs on:

- hydropower stock prices are influenced by multiple interacting factors
- raw data alone is not enough without preprocessing and feature engineering
- the study aims to determine whether hydrological and operational indicators improve market analysis

### Research Design

Mention:

- quantitative and analytical research design
- time-series and feature-based analysis
- comparative company analysis for `BARUN`, `CHCL`, and `RADHI`

### Type of Data Used

- secondary data
- rainfall and production data
- stock price-volume data
- company fundamentals

### Data Collection Method

Mention source-wise collection from files in `Datasets/`.

### Data Collection Instrument

- CSV datasets
- Python preprocessing scripts
- visualization and ML outputs

### Sample Size

Describe based on observation periods and companies included.

### Sampling Technique

- purposive sampling of hydropower companies

### Data Analysis Tool

- Python
- pandas
- matplotlib / seaborn if used
- machine learning models

### Put These Images Here

1. Raw combined rainfall-production dataset screenshot  
   Source: `Datasets/rainfall&production.csv`

2. Fundamentals dataset screenshot  
   Source: `Datasets/Fundamentals/RADHI.csv` or similar

3. Stock price dataset screenshot  
   Source: `Datasets/Price Volume/RADHI/CLOSE.csv` or similar

4. Optional processing flowchart  
   Source: `Output/ALL_00_pipeline_diagram.png`

5. Optional feature dataset screenshot  
   Source: `Features/HydroMarket/RADHI_hydro_market_features.csv`

Keep only 3-5 important visuals here.

---

## CHAPTER 4: DATA ANALYSIS, RESULTS, AND INTERPRETATION

This is the main chapter for charts and model outputs.

### Section A: Exploratory Analysis

Put these figures here:

- `Output/RP_01_production_trends.png`
- `Output/RP_02_rainfall_trends.png`
- `Output/RP_03_rainfall_vs_production.png`
- `Output/RP_04_company_generation_rainfall.png`
- `Output/RP_05_feature_correlation.png`

### Section B: Company-Level Feature Analysis

Put these figures here:

- `Output/BARUN_01_overview.png`
- `Output/BARUN_04_feature_correlation.png`
- `Output/BARUN_09_feature_heatmap.png`
- `Output/CHCL_01_overview.png`
- `Output/CHCL_04_feature_correlation.png`
- `Output/CHCL_09_feature_heatmap.png`
- `Output/RADHI_01_overview.png`
- `Output/RADHI_04_feature_correlation.png`
- `Output/RADHI_09_feature_heatmap.png`

### Section C: Overall Statistical Analysis

Put these figures here:

- `Output/ALL_02_return_distributions.png`
- `Output/ALL_03_cross_correlation.png`
- `Output/ALL_05_data_splits.png`
- `Output/ALL_06_volatility.png`
- `Output/ALL_07_seasonal_heatmap.png`
- `Output/ALL_08_summary_statistics.png`

### Section D: Machine Learning Model Results

Put these figures here:

- `Output/HydroMarketStudy/BARUN_GradientBoosting_predictions.png`
- `Output/HydroMarketStudy/BARUN_XGBoost_predictions.png`
- `Output/HydroMarketStudy/BARUN_Regularized_LSTM_predictions.png`
- `Output/HydroMarketStudy/CHCL_GradientBoosting_predictions.png`
- `Output/HydroMarketStudy/CHCL_XGBoost_predictions.png`
- `Output/HydroMarketStudy/CHCL_Regularized_LSTM_predictions.png`
- `Output/HydroMarketStudy/RADHI_GradientBoosting_predictions.png`
- `Output/HydroMarketStudy/RADHI_XGBoost_predictions.png`
- `Output/HydroMarketStudy/RADHI_Regularized_LSTM_predictions.png`

Optional training figure:

- `Output/HydroMarketStudy/*_Regularized_LSTM_training_loss.png`

### Section E: Result Tables

Use these CSV outputs to create tables or screenshots:

- `Output/HydroMarketStudy/hydro_market_model_results_ALL.csv`
- `Output/HydroMarketStudy/hydro_market_model_results_BARUN.csv`
- `Output/HydroMarketStudy/hydro_market_model_results_CHCL.csv`
- `Output/HydroMarketStudy/hydro_market_model_results_RADHI.csv`
- `Output/HydroMarketStudy/hydro_market_relation_summary_BARUN.csv`
- `Output/HydroMarketStudy/hydro_market_relation_summary_CHCL.csv`
- `Output/HydroMarketStudy/hydro_market_relation_summary_RADHI.csv`
- `Output/quality_report.csv`

Use the best tables inside Chapter 4. Put full screenshots in Appendix.

---

## CHAPTER 5: FINDINGS AND CONCLUSION

Write:

- key relation between rainfall and production
- key relation between hydrological variables and stock behavior
- company-wise differences
- best-performing model
- overall conclusion of whether hydrological and operational factors improved analysis

Possible support:

- summarize from `hydro_market_model_results_ALL.csv`

---

## CHAPTER 6: RECOMMENDATIONS AND LIMITATIONS OF THE STUDY

### Recommendations

Examples:

- include more hydropower companies
- extend the time period
- include macroeconomic indicators
- include sentiment/news data
- compare deep learning with traditional models

### Limitations

Examples:

- limited company sample
- limited time range
- possible missing values in market or hydrology data
- secondary data dependency
- model results depend on feature quality

---

## BIBLIOGRAPHY

Use APA format as required.

Include:

- research papers
- books if used
- official websites only

---

## APPENDIX

Put these screenshots here:

### Raw dataset CSV screenshots

- `Datasets/rainfall&production.csv`
- `Datasets/Fundamentals/BARUN.csv`
- `Datasets/Fundamentals/CHCL.csv`
- `Datasets/Fundamentals/RADHI.csv`
- one sample each from `Datasets/Price Volume/*`

### Feature CSV screenshots

- `Features/Rainfall/*.csv`
- `Features/Production/*.csv`
- `Features/Fundamentals/*.csv`
- `Features/Price Volume/*.csv`
- `Features/HydroMarket/*.csv`

### Result CSV screenshots

- `Output/BARUN_features.csv`
- `Output/CHCL_features.csv`
- `Output/RADHI_features.csv`
- `Output/HydroMarketStudy/*.csv`

### Extra figures

Put any figure here that is useful but too large or too detailed for Chapter 4.
