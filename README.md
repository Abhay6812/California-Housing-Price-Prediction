# 🏠 California Housing Price Prediction — Regression Benchmark

A machine learning regression project that predicts median house values using the California Housing dataset. This project benchmarks **Linear Regression (OLS), Ridge Regression, and Lasso Regression** to understand the impact of regularization, feature engineering, and hyperparameter tuning on regression performance.

---

## 🎯 Project Objective

Develop and evaluate regression models to predict median house values based on demographic, housing, and geographical features.

The project explores how regularization techniques such as Ridge (L2) and Lasso (L1) compare against standard Ordinary Least Squares (OLS) Linear Regression.

---

## 📊 Dataset

The project uses the **California Housing dataset** from Scikit-Learn.

* **Total records:** 20,640
* **Original input features:** 8
* **Engineered features:** 2
* **Total features used:** 10
* **Target variable:** `MedHouseVal`

### Features

| Feature             | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `MedInc`            | Median income in the block group                       |
| `HouseAge`          | Median house age                                       |
| `AveRooms`          | Average rooms per household                            |
| `AveBedrms`         | Average bedrooms per household                         |
| `Population`        | Block group population                                 |
| `AveOccup`          | Average household occupancy                            |
| `Latitude`          | Geographic latitude                                    |
| `Longitude`         | Geographic longitude                                   |
| `RoomsPerHousehold` | Engineered ratio of average rooms to average occupancy |
| `BedroomsPerRoom`   | Engineered ratio of average bedrooms to average rooms  |

**Target:** `MedHouseVal` — Median house value, expressed in units of $100,000.

---

## ⚙️ Machine Learning Workflow

The project follows an end-to-end regression workflow:

1. Load the California Housing dataset.
2. Perform exploratory data analysis (EDA).
3. Analyze feature distributions and target distribution.
4. Explore correlation between features and house values.
5. Engineer new features:

   * `RoomsPerHousehold`
   * `BedroomsPerRoom`
6. Separate features (`X`) and target (`y`).
7. Split data into training and testing sets (80/20).
8. Handle missing values using median imputation.
9. Investigate multicollinearity.
10. Standardize features using `StandardScaler`.
11. Train Linear Regression (OLS), Ridge, and Lasso.
12. Tune Ridge and Lasso using `GridSearchCV` with 5-fold cross-validation.
13. Evaluate models using MAE, RMSE, and R².
14. Visualize actual vs. predicted values.
15. Perform residual analysis.
16. Compare model performance.

---

## 🤖 Models Implemented

### 1. Linear Regression (OLS)

Used as the baseline model to establish initial regression performance.

### 2. Ridge Regression (L2 Regularization)

Adds a penalty to the squared coefficients to reduce coefficient magnitude and help manage multicollinearity.

### 3. Lasso Regression (L1 Regularization)

Adds a penalty based on absolute coefficient values and can shrink coefficients to zero, enabling feature selection.

### Hyperparameter Tuning

`GridSearchCV` was used with 5-fold cross-validation and RMSE scoring.

| Model | Alpha Values Tested             | Best Alpha |
| ----- | ------------------------------- | ---------: |
| Ridge | 0.001, 0.01, 0.1, 1, 10, 100    |         10 |
| Lasso | 0.0001, 0.001, 0.01, 0.1, 1, 10 |      0.001 |

---

## 📈 Model Performance

### Test Set Results

| Model                   |    MAE |       RMSE |         R² |
| ----------------------- | -----: | ---------: | ---------: |
| Linear Regression (OLS) | 0.4862 | **0.6753** | **0.6519** |
| Ridge (α = 1)           | 0.4862 |     0.6754 |     0.6519 |
| Lasso (α = 0.01)        | 0.4908 |     0.6823 |     0.6447 |
| Tuned Ridge (α = 10)    | 0.4862 |     0.6754 |     0.6519 |
| Tuned Lasso (α = 0.001) | 0.4863 |     0.6757 |     0.6516 |

### 🏆 Best Performing Model

**Linear Regression (OLS)** achieved the lowest test RMSE of **0.6753** and an R² score of **0.6519**.

The regularized models performed very similarly to OLS. Hyperparameter tuning resulted in only marginal differences, indicating that regularization did not substantially improve predictive performance with the current features and data split.

---

## 📉 Evaluation Metrics

| Metric                         | Description                                                       |
| ------------------------------ | ----------------------------------------------------------------- |
| MAE (Mean Absolute Error)      | Average absolute difference between actual and predicted values   |
| RMSE (Root Mean Squared Error) | Penalizes larger prediction errors more heavily                   |
| R² Score                       | Measures the proportion of target variance explained by the model |

Because the target is expressed in units of $100,000:

* **MAE = 0.4862** corresponds to approximately **$48,620**.
* **RMSE = 0.6753** corresponds to approximately **$67,530**.

---

## 🔍 Key Findings

* `MedInc` (Median Income) had the strongest correlation with house value, approximately **0.69**.
* House values generally increased with median income.
* The dataset showed strongly correlated feature pairs, including Latitude and Longitude.
* Feature engineering introduced room-related ratios that may provide additional information.
* OLS performed marginally better than the tuned regularized models on the test set.
* Ridge and Lasso demonstrated how regularization influences model performance and coefficient behavior.

---

## 📊 Visualizations

The notebook includes:

* Median house value distribution
* Feature distribution histograms
* Correlation heatmap
* Median income vs. house value scatter plot
* Feature multicollinearity heatmap
* RMSE model comparison chart
* Actual vs. predicted values
* Residual distribution
* Residuals vs. predicted values

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn
* **Environment:** Jupyter Notebook

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Abhay6812/California-Housing-Price-Prediction.git
```

### 2. Navigate to the project folder

```bash
cd California-Housing-Price-Prediction
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the notebook

Open `California_Housing_Regression_Benchmark.ipynb` and run the cells sequentially.

---

## 📁 Project Structure

```text
California-Housing-Price-Prediction/
│
├── California_Housing_Regression_Benchmark.ipynb
├── README.md
└── requirements.txt
```

---

## 📝 Conclusion

This project explored an end-to-end regression workflow for predicting California housing values using Linear Regression, Ridge, and Lasso.

Through exploratory data analysis, feature engineering, preprocessing, multicollinearity investigation, and hyperparameter tuning, the models were trained and evaluated using MAE, RMSE, and R².

Linear Regression achieved the best test performance, while Ridge and Lasso produced very similar results. This demonstrates that regularization does not always lead to a measurable improvement and that model performance depends on the dataset, feature representation, and regularization strength.

The project strengthened my understanding of regression, regularization, model evaluation, feature engineering, and cross-validation.

---

## 👨‍💻 Author

**Abhay Solanke**

Machine Learning | Artificial Intelligence | Data Science

---

⭐ If you find this project interesting, feel free to explore the notebook.
