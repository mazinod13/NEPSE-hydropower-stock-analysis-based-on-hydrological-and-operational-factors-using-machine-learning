# Project Report Template Structure

This folder is a ready-to-follow structure based on `PROJECT FORMAT.docx` for your topic:

`NEPSE hydropower stock analysis based on hydrological and operational factors using machine learning`

Use it to organize:

- output charts and result figures
- dataset screenshots/images
- CSV table screenshots for appendix

## Suggested Report Flow

1. `01_Title_Abstract_Prelims`
2. `02_Introduction`
3. `03_Review_of_Literature`
4. `04_Objectives_and_Methodology`
5. `05_Data_Analysis_and_Results`
6. `06_Findings_and_Conclusion`
7. `07_Recommendations_and_Limitations`
8. `08_Bibliography`
9. `09_Appendix`

## Where To Put Visual Materials In The Report

### Dataset images

Put these in Chapter 3 or early Chapter 4:

- source dataset overview screenshots
- folder structure screenshots
- example rows from raw CSV files

Best place:

- `04_Objectives_and_Methodology/dataset_images/`

### Output images

Put these in Chapter 4:

- trend charts
- correlation heatmaps
- model prediction plots
- summary analysis charts

Best place:

- `05_Data_Analysis_and_Results/output_images/`

### CSV screenshots

Put these mostly in Appendix, and only show 1-2 important samples in Chapter 3 or 4:

- screenshots of raw dataset CSV
- screenshots of processed feature CSV
- screenshots of result CSV tables

Best place:

- `09_Appendix/csv_screenshots/`

## Existing Project Files You Can Reference

### Raw datasets

- `Datasets/rainfall&production.csv`
- `Datasets/Fundamentals/BARUN.csv`
- `Datasets/Fundamentals/CHCL.csv`
- `Datasets/Fundamentals/RADHI.csv`
- `Datasets/Price Volume/BARUN/*.csv`
- `Datasets/Price Volume/CHCL/*.csv`
- `Datasets/Price Volume/RADHI/*.csv`

### Engineered features

- `Features/Rainfall/*.csv`
- `Features/Production/*.csv`
- `Features/Fundamentals/*.csv`
- `Features/Price Volume/*.csv`
- `Features/HydroMarket/*.csv`

### Analysis outputs

- `Output/*.png`
- `Output/*.csv`
- `Output/HydroMarketStudy/*.png`
- `Output/HydroMarketStudy/*.csv`

## Quick Placement Guide

### Chapter 3: Research Objectives and Methodology

Add:

- image of raw rainfall and production dataset
- image of stock price CSV structure
- image of fundamentals CSV structure
- brief preprocessing workflow or pipeline figure

### Chapter 4: Data Analysis and Results

Add:

- production trend plots
- rainfall trend plots
- rainfall vs production plots
- feature correlation plots
- feature heatmaps
- model prediction result plots
- summary statistics charts

### Appendix

Add:

- screenshots of complete CSV samples
- extra result tables from CSV outputs
- large supporting figures not placed in the main report

See `REPORT_WRITING_TEMPLATE.md` for a chapter-by-chapter fill-in format.
