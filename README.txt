# 🚲 Bike Rental Demand Prediction

A machine learning project that predicts **hourly bike rental demand** using weather, time, season, and working-day information. The project includes data generation, preprocessing, exploratory data analysis, feature engineering, model training, and model comparison.

## 📌 Features

* Generates a realistic bike rental dataset
* Handles missing values and duplicate records
* Performs Exploratory Data Analysis (EDA)
* Creates time-based and weather-based features
* Uses cyclical encoding for hour, month, and weekday
* Compares multiple regression models
* Evaluates models using **MAE, RMSE, and R²**
* Saves plots and model results automatically

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## 🤖 Machine Learning Models

The following models are trained and compared:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**

The model with the **lowest RMSE** is selected as the best-performing model.

## 📊 Features Used

The prediction is based on factors such as:

* Hour
* Month
* Weekday
* Season
* Weather condition
* Temperature
* Humidity
* Windspeed
* Working day
* Holiday
* Rush hour
* Comfort index

Cyclical features are created for **hour, month, and weekday** to represent their repeating patterns.

## 📈 Evaluation Metrics

* **MAE (Mean Absolute Error):** Average prediction error
* **RMSE (Root Mean Squared Error):** Measures prediction error with greater penalty for large errors
* **R² Score:** Measures how well the model explains the variation in rental demand

## 📁 Project Structure

```text
Bike_Rental_Prediction/
│
├── data/
│   ├── bike_rentals_dataset.csv
│   └── model_results.csv
│
├── plots/
│   ├── 01_target_distribution.png
│   ├── 02_hourly_pattern.png
│   ├── 03_season_boxplot.png
│   ├── 04_weather_boxplot.png
│   ├── 05_temp_scatter.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_model_comparison.png
│   ├── 08_actual_vs_predicted.png
│   └── 09_feature_importance.png
│
└── bike_rental_prediction.py
```

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. Set the project path

Update `BASE_DIR` in the Python script:

```python
BASE_DIR = r"C:\path\to\Bike_Rental_Prediction"
```

### 3. Run the script

```bash
python bike_rental_prediction.py
```

The dataset, plots, and model comparison results will automatically be saved inside the project folders.

## 📊 Output

The project generates:

* Bike rental dataset
* EDA visualizations
* Model performance comparison
* Actual vs. predicted plot
* Feature importance plot
* `model_results.csv`

## 🎯 Objective

The main objective is to understand the factors affecting bike rental demand and build a machine learning model that can accurately predict rental counts based on historical and environmental conditions.
