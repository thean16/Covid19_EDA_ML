# COVID-19 EDA & Death Predictor

A machine learning web app that predicts COVID-19 death counts by country using Gradient Boosting Regression, built with Streamlit.

---

## Live Demo

Run locally with:
```bash
streamlit run app.py
```

---

## Project Overview

This project performs full exploratory data analysis (EDA) on a global COVID-19 dataset and trains several regression models to predict the number of deaths per country. The best-performing model is deployed as an interactive Streamlit web application.

---

## Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Gradient Boosting** | **1,243** | **2,497** | **0.9546** |
| Random Forest | 1,508 | 3,584 | 0.9065 |
| Ridge Regression | 3,389 | 6,760 | 0.6674 |
| Linear Regression | 3,448 | 6,956 | 0.6478 |

---

## Dataset

- **Source:** `covid_19.csv`
- **Records:** 238 countries / territories
- **Date:** June 2024
- **Features:** Country, Continent, Population, Cases, Recovered, Deaths, Tests

---

## Pipeline

### 1. Data Cleaning
- Dropped rows with missing `continent` or `population`
- Filled missing `Deaths`, `Recovered`, `Tests` with `0`
- Applied **KNN Imputation** (k=3) on numeric features

### 2. Outlier Handling
- IQR-based capping (±1.5×IQR) on `Cases`, `Deaths`, `Recovered`, `Tests`

### 3. Feature Engineering
| Feature | Description |
|---|---|
| `cases_per_million` | Cases normalized by population |
| `tests_per_million` | Tests normalized by population |
| `mortality_rate` | Deaths / Cases |
| `recovery_rate` | Recovered / Cases |
| `test_positivity` | Cases / Tests |
| `continent_encoded` | Label-encoded continent |

### 4. Preprocessing
- `StandardScaler` for feature scaling
- 80/20 train-test split (`random_state=42`)

### 5. Models Trained
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor — best model, saved and deployed

---

## App Features

- Input: Population, Cases, Recovered, Tests, Continent
- Output: Predicted Deaths, Mortality Rate, Recovery Rate
- Risk label: Low / Moderate / High

---

## Project Structure

```
Covid19_EDA_ML/
├── app.py                       # Streamlit web app
├── covid.ipynb                  # Full EDA + model training notebook
├── retrain.py                   # Script to regenerate .pkl files
├── covid_19.csv                 # Raw dataset
├── covid_model.pkl              # Trained Gradient Boosting model
├── covid_scaler.pkl             # Fitted StandardScaler
├── continent_encoder.pkl        # Fitted LabelEncoder
├── boxplot_after_log.png        # Boxplot visualization
├── boxplot_before_after_log.png
└── .gitignore
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/thean16/Covid19_EDA_ML.git
cd Covid19_EDA_ML

# Install dependencies
pip install streamlit scikit-learn pandas numpy joblib

# Run the app
streamlit run app.py
```

To retrain the model from scratch:
```bash
python retrain.py
```

---

## Tech Stack

- **Python** 3.x
- **Streamlit** — web app
- **scikit-learn** — ML models, preprocessing
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — EDA visualizations
- **joblib** — model serialization
