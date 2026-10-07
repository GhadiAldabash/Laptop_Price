# Laptop Price Predictor

A machine learning project that predicts the price of a laptop from its specifications
(brand, type, RAM, storage, processor, display, and more). The project covers the full
pipeline — from cleaning raw messy data to training models — and ships with an interactive
Streamlit web app.

---

## Overview

The dataset is the **uncleaned laptop price dataset**, where specifications are stored as
messy text (e.g. `8GB`, `512GB SSD`, `Intel Core i5 2.3GHz`). The goal is to clean and
engineer these into usable numeric features, then train regression models to estimate price.

Three models are trained and compared:

| Model | What it is |
|-------|------------|
| Decision Tree | A single tree that splits the data on feature thresholds |
| Random Forest | An ensemble of many trees, averaged to reduce overfitting |
| Linear Regression | A linear baseline that assumes a straight-line relationship |

---

## Project structure

```
ML PROJECT/
├── app.py                        # Streamlit web app
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── ML_project.ipynb              # Notebook: full ML pipeline
├── .streamlit/
│   └── config.toml               # App theme (light pastel)
├── laptop_before_encoding.csv    # Cleaned data, before one-hot encoding
├── laptop_cleaned.csv            # Fully processed data (41 features)
├── decision_tree_model.pkl       # Saved Decision Tree model
├── random_forest_model.pkl       # Saved Random Forest model
├── linear_regression_model.pkl   # Saved Linear Regression model
├── scaler.pkl                    # Saved StandardScaler (for Linear Regression)
└── model_columns.pkl             # Feature column order for prediction
```

---

## The machine learning pipeline

The notebook (`ML_project.ipynb`) follows these steps:

1. **Data loading & first look** — shape, column types, and first rows.
2. **Exploratory data analysis (EDA)** — distributions, correlations, and relationships
   between features and price.
3. **Data cleaning** — removed a fully empty row and duplicates; filled missing weight and
   screen-size values with the median (less sensitive to outliers than the mean).
4. **Feature engineering** — split storage into HDD / SSD / Hybrid / Flash, extracted CPU
   speed and brand, and computed pixel density (PPI) from resolution and screen size.
5. **Encoding** — one-hot encoded categorical columns with `drop_first=True` to avoid the
   dummy-variable trap.
6. **Preprocessing** — applied `StandardScaler` (required for Linear Regression and PCA;
   tree models do not need scaling).
7. **Dimensionality reduction (PCA)** — applied PCA and studied the explained variance.
8. **Modeling** — trained the three regression models.
9. **Evaluation** — compared R², MAE, and MSE, plus actual-vs-predicted plots.

---

## Results

On the held-out test set (20% of the data):

| Model | R² | MAE | MSE |
|-------|-----|-----|-----|
| Decision Tree | 0.656 | 13,704 | 550.8M |
| Linear Regression | 0.744 | 13,317 | 410.1M |
| **Random Forest** | **0.805** | **10,240** | **312.0M** |

**Random Forest performed best**, as it averages many trees to reduce overfitting and
capture non-linear patterns. All models are less accurate for very high-priced laptops,
which is explained by the right-skewed price distribution and the small number of expensive
laptops in the data.

### A note on PCA

PCA needed **31 of 41 components** to retain 95% of the variance, and the cumulative-variance
curve showed no clear elbow. This suggests dimensionality reduction did not help much here —
likely because most features are independent one-hot dummy columns. The models were therefore
trained on the original features, with PCA kept for comparison.

---

## The Streamlit app

The app (`app.py`) presents the whole project interactively across five pages:

- **Overview** — project summary and a sample of the data.
- **Data Exploration** — interactive charts for price distribution, price by type and brand,
  RAM and SSD vs price, and a correlation heatmap.
- **Preprocessing** — a before/after view of the cleaning and encoding steps.
- **Model Performance** — metrics for all three models, an R² comparison, and
  actual-vs-predicted plots.
- **Predict Price** — choose a model and enter laptop specs through dropdowns and sliders to
  get a live price estimate.

---

## How to run

1. Make sure all files listed in the project structure are in the same folder.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. The app opens in your browser at `http://localhost:8501`.

---

## Built with

- **Python** — pandas, NumPy
- **scikit-learn** — modeling, scaling, PCA, metrics
- **Plotly** — interactive charts
- **Streamlit** — the web app
- **joblib** — saving and loading the trained models
