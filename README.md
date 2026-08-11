# Model_Ev_Fiyat_Tahmini_Ai_MetaLearnerPulp
# 📉 End-to-End ML/OR Portfolio Optimization Engine

This project is an **End-to-End Decision Support System (DSS)** designed to identify real estate pricing anomalies and market arbitrage by combining heterogeneous machine learning models (Stacking Regressor) with mathematical optimization techniques (Binary Knapsack Optimization via PuLP).

---

## 🛠️ Technical Architecture & Methodology

### 1. Heterogeneous Meta-Learner (Two-Stage Stacking Ensemble)
The model utilizes a two-stage Stacking architecture to optimize the bias-variance tradeoff:
* **Level-0 Base Estimators:** 
  * `LightGBM Regressor` (Gradient-based Decision Trees)
  * `Random Forest Regressor` (Bagging Ensemble)
  * `Gradient Boosting Regressor` (Boosting Ensemble)
* **Level-1 Meta-Estimator:** 
  * `LGBMRegressor` (Final decision estimator that generalizes the meta-feature space)
* **Data Leakage Prevention:** Out-of-fold prediction via $K$-Fold Cross-Validation is implemented across the pipeline to eliminate data leakage and prevent overfitting.

### 2. Operations Research & Knapsack Formulation
The marginal expected profit ($\Delta_i$) per asset is computed as the spread between the Meta-Learner's predicted fair market value ($\hat{Y}_i$) and the current market listing price ($P_i$):

$$\Delta_i = \hat{Y}_i - P_i$$

The portfolio allocation problem is formulated as a **Binary Integer Linear Programming (BILP)** model:

* **Decision Variable:** 
  $$x_i \in \{0, 1\} \quad (x_i = 1 \text{ if the asset is selected for the portfolio, } 0 \text{ otherwise})$$

* **Objective Function (Profit Maximization):**
  $$\max \sum_{i=1}^{N} x_i \cdot \Delta_i$$

* **Capital Constraint:**
  $$\sum_{i=1}^{N} x_i \cdot P_i \le B$$
  *(Where $B$ represents the defined total investment capital budget.)*

---

## 🔬 System Pipeline Architecture
