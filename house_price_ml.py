# ============================================================
#  HomeIQ – House Price Prediction (ML Project)
#  Author  : Your Name | 3rd Year CS Student
#  Stack   : Python, Scikit-learn, Pandas, Flask
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. GENERATE / LOAD DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
n = 2000

data = pd.DataFrame({
    'area_sqft'  : np.random.randint(500, 4000, n),
    'bedrooms'   : np.random.randint(1, 7, n),
    'bathrooms'  : np.random.randint(1, 5, n),
    'age_years'  : np.random.randint(0, 40, n),
    'garage'     : np.random.choice([0, 1], n),
    'location'   : np.random.choice(['Prime', 'Suburban', 'Rural'], n,
                                    p=[0.25, 0.55, 0.20]),
})

loc_map   = {'Prime': 1.45, 'Suburban': 1.0, 'Rural': 0.72}
age_decay = np.maximum(0.75, 1 - data['age_years'] * 0.012)
noise     = np.random.normal(0, 0.05, n)

data['price'] = (
    data['area_sqft'] * 95
    + data['bedrooms']  * 8000
    + data['bathrooms'] * 5500
    + data['garage']    * 12000
) * data['location'].map(loc_map) * age_decay * (1 + noise)

data['price'] = data['price'].round(-3).astype(int)

print("=" * 55)
print("  HomeIQ — House Price Prediction")
print("=" * 55)
print(f"\n📦 Dataset: {data.shape[0]} rows × {data.shape[1]} columns")
print(f"💰 Price range: ${data['price'].min():,} – ${data['price'].max():,}")
print(f"📊 Avg price:   ${data['price'].mean():,.0f}")
print(f"\n{data.head().to_string()}")

# ── Missing value check ──────────────────────
print(f"\n🔍 Missing values:\n{data.isnull().sum()}")
print(f"\n📈 Basic Stats:\n{data.describe().round(2)}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
#    (adds value to your project — shows you
#     know more than just calling .fit())
# ─────────────────────────────────────────────
data['price_per_sqft']   = (data['price'] / data['area_sqft']).round(2)
data['bed_bath_ratio']   = (data['bedrooms'] / data['bathrooms']).round(2)
data['total_rooms']      = data['bedrooms'] + data['bathrooms']
data['is_new']           = (data['age_years'] <= 5).astype(int)
data['area_x_location']  = data['area_sqft'] * data['location'].map(loc_map)

print("\n✅ Feature engineering complete — 5 new features added")
print(f"   New shape: {data.shape}")

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
fig = plt.figure(figsize=(18, 12))
fig.suptitle("HomeIQ — Exploratory Data Analysis", fontsize=18,
             fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. Price Distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(data['price']/1000, bins=45, color='#6c63ff', edgecolor='white',
         alpha=0.85, linewidth=0.4)
ax1.axvline(data['price'].mean()/1000, color='#ff6584', lw=2,
            linestyle='--', label=f"Mean: ${data['price'].mean()/1000:.0f}K")
ax1.set_title("Price Distribution", fontweight='bold')
ax1.set_xlabel("Price ($K)"); ax1.set_ylabel("Count")
ax1.legend(fontsize=8)

# 2. Area vs Price (coloured by location)
ax2 = fig.add_subplot(gs[0, 1])
colours = {'Prime':'#6c63ff','Suburban':'#43e97b','Rural':'#ff6584'}
for loc, grp in data.groupby('location'):
    ax2.scatter(grp['area_sqft'], grp['price']/1000,
                alpha=0.25, s=8, color=colours[loc], label=loc)
ax2.set_title("Area vs Price (by Location)", fontweight='bold')
ax2.set_xlabel("Area (sq ft)"); ax2.set_ylabel("Price ($K)")
ax2.legend(fontsize=8, markerscale=2)

# 3. Boxplot: bedrooms vs price
ax3 = fig.add_subplot(gs[0, 2])
sns.boxplot(data=data, x='bedrooms', y='price', ax=ax3,
            palette='viridis', flierprops=dict(marker='o', markersize=2))
ax3.set_title("Bedrooms vs Price", fontweight='bold')
ax3.set_ylabel("Price ($)")

# 4. Location average price
ax4 = fig.add_subplot(gs[1, 0])
loc_avg = data.groupby('location')['price'].mean().sort_values(ascending=False)
bars = ax4.bar(loc_avg.index, loc_avg.values/1000,
               color=['#6c63ff','#43e97b','#ff6584'], edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, loc_avg.values/1000):
    ax4.text(bar.get_x()+bar.get_width()/2, v+2, f'${v:.0f}K',
             ha='center', fontsize=9, fontweight='bold')
ax4.set_title("Avg Price by Location", fontweight='bold')
ax4.set_ylabel("Avg Price ($K)")

# 5. Correlation Heatmap
ax5 = fig.add_subplot(gs[1, 1])
le_tmp = LabelEncoder()
tmp = data.copy()
tmp['location'] = le_tmp.fit_transform(tmp['location'])
corr_cols = ['area_sqft','bedrooms','bathrooms','age_years',
             'garage','location','price']
corr = tmp[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', ax=ax5, cmap='coolwarm',
            linewidths=0.5, mask=mask, vmin=-1, vmax=1,
            annot_kws={"size":8})
ax5.set_title("Correlation Heatmap", fontweight='bold')

# 6. Age vs Price
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(data['age_years'], data['price']/1000,
            alpha=0.2, c='#ffd166', s=8)
# trend line
z = np.polyfit(data['age_years'], data['price']/1000, 1)
p = np.poly1d(z)
x_line = np.linspace(0, 40, 100)
ax6.plot(x_line, p(x_line), color='#ff6584', lw=2, label='Trend')
ax6.set_title("Age vs Price", fontweight='bold')
ax6.set_xlabel("Age (years)"); ax6.set_ylabel("Price ($K)")
ax6.legend(fontsize=8)

plt.savefig("eda_plots.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n📊 EDA plots saved → eda_plots.png")

# ─────────────────────────────────────────────
# 4. PREPROCESSING
# ─────────────────────────────────────────────
le = LabelEncoder()
df = data.copy()
df['location'] = le.fit_transform(df['location'])

# Drop derived columns not used in model training
# (keep engineered features for better accuracy)
feature_cols = ['area_sqft','bedrooms','bathrooms','age_years',
                'garage','location','bed_bath_ratio','total_rooms',
                'is_new','area_x_location']

X = df[feature_cols]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n📂 Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")
print(f"🔢 Features used: {len(feature_cols)}")

# ─────────────────────────────────────────────
# 5. MODEL TRAINING & CROSS-VALIDATION
#    Cross-validation shows you understand
#    how to properly evaluate a model
# ─────────────────────────────────────────────
models = {
    "Linear Regression"  : LinearRegression(),
    "Ridge Regression"   : Ridge(alpha=10.0),
    "Random Forest"      : RandomForestRegressor(
                               n_estimators=200, max_depth=12,
                               min_samples_leaf=3, random_state=42, n_jobs=-1),
    "Gradient Boosting"  : GradientBoostingRegressor(
                               n_estimators=200, learning_rate=0.08,
                               max_depth=5, subsample=0.85,
                               random_state=42),
}

print("\n" + "=" * 65)
print(f"{'Model':<22} {'R²':>7} {'MAE':>12} {'RMSE':>12} {'CV R² (5-fold)':>16}")
print("=" * 65)

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    use_scaled = name in ("Linear Regression", "Ridge Regression")
    X_tr = X_train_sc if use_scaled else X_train
    X_te = X_test_sc  if use_scaled else X_test

    model.fit(X_tr, y_train)
    preds = model.predict(X_te)

    r2   = r2_score(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # 5-fold cross-validation on training data
    X_cv = X_train_sc if use_scaled else X_train
    cv_scores = cross_val_score(model, X_cv, y_train,
                                cv=kf, scoring='r2', n_jobs=-1)

    results[name] = {
        "R2": r2, "MAE": mae, "RMSE": rmse,
        "CV_mean": cv_scores.mean(), "CV_std": cv_scores.std(),
        "model": model, "preds": preds
    }
    print(f"{name:<22} {r2:>7.4f} ${mae:>10,.0f} ${rmse:>10,.0f}"
          f"  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("=" * 65)

best_name  = max(results, key=lambda k: results[k]["R2"])
best_model = results[best_name]["model"]
print(f"\n🏆 Best Model: {best_name}")
print(f"   R² = {results[best_name]['R2']:.4f}")
print(f"   MAE = ${results[best_name]['MAE']:,.0f}")
print(f"   CV  = {results[best_name]['CV_mean']:.4f} ± {results[best_name]['CV_std']:.4f}")

# ─────────────────────────────────────────────
# 6. RESULTS VISUALISATION
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("HomeIQ — Model Evaluation", fontsize=15, fontweight='bold')

# R² Comparison
names = list(results.keys())
r2s   = [results[n]["R2"] for n in names]
short = ["Linear\nReg.", "Ridge\nReg.", "Random\nForest", "Gradient\nBoosting"]
bar_colours = ['#ff6584','#ffd166','#6c63ff','#43e97b']
bars = axes[0].bar(short, r2s, color=bar_colours, edgecolor='white', linewidth=0.5)
axes[0].set_ylim(0.80, 1.02)
axes[0].set_title("R² Score Comparison", fontweight='bold')
axes[0].set_ylabel("R² Score")
for bar, v in zip(bars, r2s):
    axes[0].text(bar.get_x()+bar.get_width()/2, v+0.003,
                 f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# Actual vs Predicted
preds_best = results[best_name]["preds"]
axes[1].scatter(y_test/1000, preds_best/1000,
                alpha=0.35, s=8, color='#6c63ff')
mn, mx = y_test.min()/1000, y_test.max()/1000
axes[1].plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect fit')
residuals_pct = ((preds_best - y_test) / y_test * 100).abs().mean()
axes[1].set_xlabel("Actual Price ($K)")
axes[1].set_ylabel("Predicted Price ($K)")
axes[1].set_title(f"Actual vs Predicted\n({best_name})", fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].text(0.05, 0.92, f"Avg Error: {residuals_pct:.1f}%",
             transform=axes[1].transAxes, fontsize=9, color='white',
             bbox=dict(boxstyle='round', facecolor='#6c63ff', alpha=0.7))

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_,
                    index=X.columns).sort_values(ascending=True)
    colours_imp = ['#6c63ff' if v > imp.median() else '#aaa' for v in imp.values]
    imp.plot(kind='barh', ax=axes[2], color=colours_imp)
    axes[2].set_title("Feature Importance", fontweight='bold')
    axes[2].set_xlabel("Importance Score")
    for i, (idx, v) in enumerate(imp.items()):
        axes[2].text(v+0.001, i, f'{v:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig("model_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("📊 Results saved → model_results.png")

# ─────────────────────────────────────────────
# 7. RESIDUAL ANALYSIS
#    Shows you understand model diagnostics —
#    this is what separates good ML work
# ─────────────────────────────────────────────
residuals = y_test - preds_best

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Residual Analysis — " + best_name, fontweight='bold')

# Residuals vs Predicted
axes[0].scatter(preds_best/1000, residuals/1000,
                alpha=0.3, s=8, color='#6c63ff')
axes[0].axhline(0, color='red', lw=1.5, linestyle='--')
axes[0].set_xlabel("Predicted Price ($K)")
axes[0].set_ylabel("Residual ($K)")
axes[0].set_title("Residuals vs Predicted")

# Residual distribution
axes[1].hist(residuals/1000, bins=40, color='#43e97b',
             edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', lw=1.5, linestyle='--')
axes[1].set_xlabel("Residual ($K)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Residual Distribution\n"
                  f"(mean={residuals.mean()/1000:.1f}K, "
                  f"std={residuals.std()/1000:.1f}K)")

plt.tight_layout()
plt.savefig("residuals.png", dpi=150, bbox_inches='tight')
plt.show()
print("📊 Residual plot saved → residuals.png")

# ─────────────────────────────────────────────
# 8. PREDICT NEW HOUSE
# ─────────────────────────────────────────────
def predict_price(area, beds, baths, age, garage, location_label):
    """
    Predict house price using the best trained model.
    
    Parameters:
        area (int)           : Floor area in sq ft
        beds (int)           : Number of bedrooms
        baths (int)          : Number of bathrooms
        age (int)            : Age of property in years
        garage (int)         : 1 = has garage, 0 = no garage
        location_label (str) : 'Prime', 'Suburban', or 'Rural'
    
    Returns:
        float: Predicted price in USD
    """
    loc_enc_map = {'Prime': 1, 'Suburban': 2, 'Rural': 0}
    loc_mult    = {'Prime': 1.45, 'Suburban': 1.0, 'Rural': 0.72}

    bed_bath_ratio  = beds / baths
    total_rooms     = beds + baths
    is_new          = int(age <= 5)
    area_x_location = area * loc_mult[location_label]

    inp = pd.DataFrame([[
        area, beds, baths, age, garage,
        loc_enc_map[location_label],
        bed_bath_ratio, total_rooms, is_new, area_x_location
    ]], columns=feature_cols)

    price = best_model.predict(inp)[0]
    conf_low  = price * 0.92
    conf_high = price * 1.08

    print(f"\n{'─'*45}")
    print(f"  🏠  {beds}bed/{baths}bath | {area:,} sqft | {location_label}")
    print(f"  Age: {age} yrs | Garage: {'Yes' if garage else 'No'}")
    print(f"{'─'*45}")
    print(f"  💰  Estimated Price : ${price:>12,.0f}")
    print(f"  📉  Low  (−8%)      : ${conf_low:>12,.0f}")
    print(f"  📈  High (+8%)      : ${conf_high:>12,.0f}")
    print(f"  📐  Price / sq ft   : ${price/area:>12,.0f}")
    print(f"{'─'*45}")
    return price


print("\n" + "="*45)
print("  SAMPLE PREDICTIONS")
print("="*45)
predict_price(area=1800, beds=3, baths=2, age=8,  garage=1, location_label='Suburban')
predict_price(area=2500, beds=4, baths=3, age=3,  garage=1, location_label='Prime')
predict_price(area=900,  beds=2, baths=1, age=25, garage=0, location_label='Rural')

print("\n✅ HomeIQ pipeline complete!")
print("   Outputs: eda_plots.png | model_results.png | residuals.png")
