<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=32&duration=2800&pause=2000&color=1A56DB&center=true&vCenter=true&width=600&lines=🏠+HomeIQ;AI+House+Price+Predictor;Built+by+Nalla+Tarun+Sai" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

<br/>

### 🎯 92.4% R² Accuracy &nbsp;·&nbsp; 4 ML Models &nbsp;·&nbsp; 5-Fold CV &nbsp;·&nbsp; Live Demo

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-1A56DB?style=for-the-badge)](https://TarunSaiNalla.github.io/homeiq)
[![GitHub](https://img.shields.io/badge/⭐_Star_This_Repo-TarunSaiNalla-black?style=for-the-badge&logo=github)](https://github.com/TarunSaiNalla/homeiq)

<br/>

> *"The best investment on Earth is earth." — Louis Glickman*

---

</div>

## 📌 What is HomeIQ?

**HomeIQ** is a complete end-to-end Machine Learning project that predicts residential property prices using ensemble algorithms. Built as a 3rd year CS project, it covers the full ML lifecycle — from data generation and EDA to feature engineering, model comparison, cross-validation, residual analysis, and a live interactive web demo.

No backend needed. Open the HTML file in any browser and it just works.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏠 **Instant Prediction** | Estimates market value in real time |
| 📊 **4 Models Compared** | Random Forest, Gradient Boosting, Ridge, Linear Regression |
| ⚙️ **Feature Engineering** | 5 custom derived features for higher accuracy |
| ✅ **5-Fold Cross-Validation** | Proper evaluation — not just a lucky train/test split |
| 🔍 **Residual Analysis** | Full diagnostic plots to understand model errors |
| 🌐 **Zero Backend** | Runs 100% in the browser — no server needed |
| 💡 **Price Breakdown** | Shows exactly how each feature contributes to the price |
| 🎨 **Beautiful UI** | Responsive, modern frontend with animated results |

---

## 🧠 Model Performance

<div align="center">

| # | Model | CV R² | Test R² | MAE | RMSE |
|---|---|---|---|---|---|
| 🥇 | **Random Forest** | 92.18% ± 0.0019 | **92.4%** | $12,400 | $18,200 |
| 🥈 | Gradient Boosting | 91.62% ± 0.0022 | 91.8% | $13,100 | $19,500 |
| 🥉 | Ridge Regression | 87.01% ± 0.0028 | 87.1% | $21,600 | $28,900 |
| 4️⃣ | Linear Regression | 86.98% ± 0.0031 | 87.1% | $21,800 | $29,100 |

</div>

> ✅ **Random Forest** selected as best model — highest R² and most stable CV scores.

---

## 🏗️ Features Used (10 Total)

```
📐 Raw Features (6)          ⚙️ Engineered Features (4)
─────────────────────        ──────────────────────────
• Area (sq ft)               • Bed-Bath Ratio
• Bedrooms                   • Total Rooms
• Bathrooms                  • Is New (age ≤ 5 yrs)
• Age (years)                • Area × Location Multiplier
• Garage (yes/no)
• Location Tier
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/TarunSaiNalla/homeiq.git
cd homeiq

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Train the model + generate all plots
python homeiq_model.py

# 4. Open the live demo
# Just open index.html in your browser — done!
```

---

## 🔬 How It Works

```
User Input → Feature Engineering → StandardScaler → Random Forest → Price + Confidence Range
    │                │                                    │
    │         5 new features                         ±8% range
    │         derived live                           breakdown
    │                                                shown in UI
    └── area, beds, baths, age, garage, location
```

**Step by step:**
1. 🖊️ User enters property details — area, rooms, age, location, garage
2. ⚙️ 5 engineered features derived automatically
3. 🤖 Random Forest predicts price with confidence range
4. 💡 Full breakdown shown — base value, bonuses, location multiplier, age decay
5. 🎨 Result animates into the UI instantly

---

## 📊 Dataset

```
Total Samples     : 2,000
Train / Test      : 80% / 20%  (fixed seed = 42)
Cross-Validation  : 5-Fold Stratified
Noise             : np.random.normal(0, 0.05)
Price Range       : ~$54,000 – $612,000
```

---

## 📁 Project Structure

```
homeiq/
│
├── 🐍 homeiq_model.py     ← ML pipeline: train, evaluate, predict
├── 🌐 index.html          ← Live interactive frontend demo
│
├── 📊 eda_plots.png        ← 6-panel EDA visualisation
├── 📈 model_results.png   ← R², actual vs predicted, feature importance
├── 🔍 residuals.png        ← Residual analysis plots
│
└── 📄 README.md
```

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
![GitHub Pages](https://img.shields.io/badge/GitHub_Pages-181717?style=flat-square&logo=github&logoColor=white)

</div>

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

## 👨‍💻 Author

**Nalla Tarun Sai**
3rd Year Computer Science Student

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tarun-sai-nalla-95a1103a2)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/TarunSaiNalla)

<br/>

⭐ **If you found this helpful, please star the repo!** ⭐

<br/>

*Built with ❤️ by Nalla Tarun Sai · 2025*

</div>
