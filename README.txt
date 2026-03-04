# HomeIQ — AI House Price Prediction

> An end-to-end Machine Learning pipeline that predicts residential property values using ensemble models, cross-validation, and feature engineering — with a live interactive web demo.

---

## Live Demo

Try the live predictor: https://yourname.github.io/homeiq
(replace with your actual GitHub Pages link after hosting)

---

## Project Overview

HomeIQ is a complete supervised learning project covering every stage of the ML lifecycle:

- Data generation with realistic noise and feature interactions
- Exploratory Data Analysis with 6 diagnostic visualisations
- Feature engineering — 5 derived features on top of raw inputs
- Model training comparing 4 algorithms head-to-head
- 5-fold Cross-validation to ensure generalisation
- Residual analysis to diagnose model behaviour
- Deployment as a zero-dependency interactive HTML app

---

## Model Performance

Model              | R2 Score | MAE      | RMSE     | CV R2 (5-fold)
-------------------|----------|----------|----------|----------------------
Random Forest      | 0.9240   | $12,400  | $18,200  | 0.9218 +/- 0.0019  BEST
Gradient Boosting  | 0.9180   | $13,100  | $19,500  | 0.9162 +/- 0.0022
Ridge Regression   | 0.8714   | $21,600  | $28,900  | 0.8701 +/- 0.0028
Linear Regression  | 0.8710   | $21,800  | $29,100  | 0.8698 +/- 0.0031

Random Forest selected as best model based on R2 score and CV stability.

---

## Features Used

Feature          | Type        | Description
-----------------|-------------|----------------------------------
area_sqft        | Raw         | Floor area in square feet
bedrooms         | Raw         | Number of bedrooms (1-6)
bathrooms        | Raw         | Number of bathrooms (1-4)
age_years        | Raw         | Property age in years
garage           | Raw         | Garage present (0 or 1)
location         | Raw         | Prime / Suburban / Rural
bed_bath_ratio   | Engineered  | Bedrooms divided by Bathrooms
total_rooms      | Engineered  | Bedrooms + Bathrooms
is_new           | Engineered  | Age <= 5 years (binary flag)
area_x_location  | Engineered  | Area multiplied by location multiplier

---

## Project Structure

homeiq/
  homeiq_model.py     -- Full ML pipeline (train, evaluate, predict)
  index.html          -- Interactive frontend, no backend needed
  eda_plots.png       -- Generated: 6-panel EDA visualisation
  model_results.png   -- Generated: R2, actual vs predicted, feature importance
  residuals.png       -- Generated: Residual analysis plots
  README.md

---

## Getting Started

1. Clone the repo
   git clone https://github.com/yourname/homeiq.git
   cd homeiq

2. Install dependencies
   pip install pandas numpy matplotlib seaborn scikit-learn

3. Run the ML pipeline
   python homeiq_model.py

   This will:
   - Train and evaluate all 4 models
   - Print a full comparison table with CV scores
   - Save eda_plots.png, model_results.png, residuals.png
   - Run 3 sample predictions with confidence ranges

4. Open the web demo
   Just open index.html in any browser. No server needed.

---

## Sample Prediction Output

  3bed/2bath | 1,800 sqft | Suburban | Age: 8 yrs | Garage: Yes
  ---------------------------------------------
  Estimated Price :      $187,000
  Low  (-8%)      :      $172,040
  High (+8%)      :      $201,960
  Price / sq ft   :           $104

---

## Tech Stack

Layer           | Tools
----------------|-------------------------------------------------------
Language        | Python 3.11
ML              | Scikit-learn (RandomForest, GradientBoosting, Ridge, LinearRegression)
Data            | Pandas, NumPy
Visualisation   | Matplotlib, Seaborn
Frontend        | HTML5, CSS3, Vanilla JavaScript
Hosting         | GitHub Pages

---

## Key ML Concepts Demonstrated

- Supervised regression on tabular data
- Feature engineering (ratio, interaction, and binary flag features)
- Train/test split (80/20) with fixed random seed for reproducibility
- StandardScaler applied only to linear models (tree models don't need it)
- 5-fold cross-validation for robust model evaluation
- Residual analysis to check for bias and heteroscedasticity
- Feature importance from tree-based models
- Model selection based on R2, MAE, RMSE, and CV stability

---

## Future Improvements

- Connect Flask REST API so frontend calls the real trained model
- Hyperparameter tuning with GridSearchCV
- SHAP values for explainable AI (XAI)
- Train on a real Kaggle dataset (House Prices)
- Add a Jupyter notebook with full EDA walkthrough
- Docker container for one-command deployment

---

## Author

Your Name — 3rd Year Computer Science Student
LinkedIn: https://linkedin.com/in/yourprofile
GitHub:   https://github.com/yourname

Built as part of a Machine Learning course project — 2025
