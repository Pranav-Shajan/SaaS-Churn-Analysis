# SaaS Customer Churn Analysis

## Overview
End-to-end churn analysis on 7,043 SaaS subscribers using SQL, 
Python, and Power BI to identify high-risk customers and support 
retention decisions.

## Key Findings
- Month-to-month customers churn at 42.7% vs 2.8% for two-year contracts
- First-year customers show a 47.7% churn rate
- 931 high-risk customers identified averaging $81/month in revenue
- Random Forest model achieved ROC-AUC of 0.84

## SQL Analysis Results

| Query | Finding |
|---|---|
| Overall churn rate | 26.6% of 7,043 subscribers churned |
| Churn by contract | Month-to-month: 42.7% vs Two year: 2.8% |
| Churn by tenure | First year: 47.7%, drops to 9.5% after 4 years |
| High risk segment | 931 customers, avg $81/mo, tenure < 12 months |
| Revenue at risk | $139K monthly revenue from churned customers |

See `/sql_outputs` folder for query screenshots.

## Tools Used
- **SQL** — Exploratory analysis (CTEs, conditional aggregation, joins)
- **Python** — Data cleaning, EDA, Random Forest model (pandas, scikit-learn)
- **Power BI** — Interactive dashboard with churn predictions

## Project Structure
- `churn_analysis.sql` — SQL queries for exploratory analysis
- `churn_analysis.py` — Python script for EDA and ML model
- `churn_predictions.csv` — Model predictions exported for Power BI
- `Churn_Analysis.pbix` — Power BI dashboard
- `churn_eda.png` — EDA charts
- `feature_importance.png` — Top 10 churn predictors

## Dataset
IBM Telco Customer Churn dataset — available on Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## How to Run
1. Download dataset from Kaggle link above
2. Place in project folder and run `churn_analysis.py`
3. Open `Churn_Analysis.pbix` in Power BI Desktop
