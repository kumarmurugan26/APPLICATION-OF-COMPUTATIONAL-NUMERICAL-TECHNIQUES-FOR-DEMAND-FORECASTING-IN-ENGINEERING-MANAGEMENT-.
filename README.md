# Demand Forecasting for Supply Chain Planning

Applying regression-based numerical methods to forecast product demand 
using pricing, promotion, economic, and competitor variables — built to 
support inventory, pricing, and planning decisions.

## Business problem
Poor demand forecasts drive overstocking and stockouts. This project 
tests whether simple, interpretable models (linear regression) can 
match more complex ones (polynomial regression) for demand prediction — 
a tradeoff every supply chain planner has to make between accuracy and 
usability.

## Approach
- Custom linear regression via the Normal Equation (from scratch, NumPy)
- Built-in scikit-learn linear regression (validation baseline)
- Polynomial regression (complexity comparison)
- Evaluated with RMSE and MAE

## Key finding
Linear regression matched polynomial regression in accuracy (RMSE 54.91 
vs 55.38) while being simpler and more interpretable — supporting the 
case for using the simplest model that does the job in operational 
forecasting.

## Tech stack
Python · pandas · NumPy · scikit-learn · matplotlib

## Dataset
[Strategic Supply Chain Demand Forecasting Dataset (Kaggle)](https://www.kaggle.com/datasets/ziya07/strategic-supply-chain-demand-forecasting-dataset)


## 📚 References

* Research papers on demand forecasting and regression models
* Engineering management numerical analysis studies

---

## 👤 Author

**Kumar Murugan**

