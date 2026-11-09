# 📊 Basic Statistical Analysis of Student Scores

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An exploratory data analysis (EDA) and descriptive statistics tool in Python using **Pandas**, **NumPy**, and **Matplotlib** to evaluate student academic performance across quantitative (Math) and verbal exam distributions.

---

## ✨ Key Features

- **Central Tendency & Dispersion:** Computes Mean ($\mu$) with **95% Confidence Intervals**, Median ($Q_2$), Mode, Sample Variance ($s^2$), Standard Deviation ($s$), Range, and Interquartile Range ($\text{IQR} = Q_3 - Q_1$).
- **Distribution Shape & Outlier Detection:** Evaluates **Skewness**, **Kurtosis**, and identifies extreme observations via Z-score thresholding ($|Z| > 2.5$).
- **Bivariate Correlation Analysis:** Calculates the **Pearson Correlation Coefficient ($r$)** and fits an Ordinary Least Squares (OLS) linear trendline between Math and Verbal scores.
- **2×2 Visual Analytics Dashboard:** Automatically renders and exports `score_distribution.png` containing:
  1. Math Score Histogram with Mean/Median overlays
  2. Verbal Score Histogram with Mean/Median overlays
  3. Side-by-Side Boxplot & Quartile comparison
  4. Bivariate Scatter Plot with Linear Regression line

---

## 🗂️ Project Structure

```text
Basic-Statistical-Analysis/
├── data/
│   └── scores.txt           # Student exam score dataset (ID, Math, Verbal)
├── main.py                  # Statistical analysis & visualization pipeline
├── requirements.txt         # Python package dependencies
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Firatdondar/Basic-Statistical-Analysis.git
cd Basic-Statistical-Analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Analysis
```bash
python main.py
```

---

## 📈 Sample Console Output

```text
========================================================================
                 DESCRIPTIVE STATISTICAL ANALYSIS REPORT
========================================================================

[MATH SCORE] (n = 99)
---------------------------------------------
  Mean ± 95% CI      : 62.66 ± 3.55
  Median (Q2)        : 62.08
  Mode               : 30.89
  Variance (s²)      : 325.15
  Std. Deviation (s) : 18.03
  Range (Min - Max)  : 30.89 - 98.41 (Span: 67.52)
  Quartiles          : Q1=47.76 | Q2=62.08 | Q3=77.38 (IQR=29.62)
  Skewness / Kurtosis: 0.089 / -1.045
  Outliers (|Z|>2.5) : 0
```

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
