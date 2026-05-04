
# 📊 Demand Forecasting using Numerical Techniques

## 📌 Project Overview

This project focuses on applying **computational numerical techniques** for **demand forecasting** in an engineering management context. The objective is to build accurate and interpretable forecasting models using regression methods and evaluate their performance using error metrics.

The project demonstrates how **data-driven decision-making** can improve operational planning such as inventory control, pricing, and resource allocation.

---

## 🚀 Key Features

* 📈 Exploratory Data Analysis (EDA)
* 📉 Linear Regression (Custom Implementation – Normal Equation)
* 🤖 Built-in Regression using Scikit-learn
* 🔁 Polynomial Regression for non-linear modeling
* 📊 Model comparison using RMSE & MAE
* 📌 Visualization of trends, correlations, and residuals

---

## 🧠 Problem Statement

Demand forecasting is critical in engineering management. Poor predictions can lead to:

* Overstocking
* Stockouts
* Financial losses

This project models demand as a function of multiple variables:

```
Future Demand = f(Price, Discount, Promotion, Economy, Competition, Weather)
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab

---

## 📂 Project Structure

```
├── data/
│   └── demand_forecasting_dataset.csv
├── notebooks/
│   └── demand_forecasting.ipynb
├── src/
│   └── regression_models.py
├── results/
│   ├── plots/
│   └── metrics/
├── README.md
```

---

## 📊 Models Implemented

### 1. Custom Linear Regression

* Implemented using **Normal Equation**
* No iterative optimization required
* Helps understand mathematical foundations

### 2. Built-in Linear Regression

* Using `sklearn.linear_model.LinearRegression`
* Efficient and numerically stable

### 3. Polynomial Regression

* Captures non-linear relationships
* Higher complexity

---

## 📉 Performance Metrics

| Model                      | RMSE  | MAE   | Complexity | Interpretability |
| -------------------------- | ----- | ----- | ---------- | ---------------- |
| Custom Linear Regression   | 54.91 | 47.50 | Low        | High             |
| Built-in Linear Regression | 54.91 | 47.50 | Low        | High             |
| Polynomial Regression      | 55.38 | 47.80 | Medium     | Medium           |

📌 **Insight:** Simpler linear models performed better than more complex polynomial models for this dataset.

---

## 📈 Visualizations

* Sales Units over Time
* Price vs Sales Scatter Plot
* Actual vs Predicted Demand
* Residual Distribution
* RMSE & MAE Comparison Charts

---

## 🔍 Key Insights

* Price had **very weak correlation** with demand
* Linear regression provided **stable and consistent results**
* Increasing model complexity did **not improve performance significantly**
* Residuals were evenly distributed → model assumptions are valid

---

## 📥 Dataset

Dataset used:
👉 [https://www.kaggle.com/datasets/ziya07/strategic-supply-chain-demand-forecasting-dataset](https://www.kaggle.com/datasets/ziya07/strategic-supply-chain-demand-forecasting-dataset)

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/demand-forecasting.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the notebook:

```
jupyter notebook
```

---

## 📌 Conclusion

This project demonstrates that:

* Numerical techniques are effective for complex forecasting problems
* Linear regression offers a strong balance between **accuracy and interpretability**
* Python-based tools significantly enhance computational efficiency and decision-making

---

## 📚 References

* Research papers on demand forecasting and regression models
* Engineering management numerical analysis studies

---

## 👤 Author

**Kumar Murugan**

