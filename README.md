# 🚲 Bike Rental Demand Prediction

A machine learning project that predicts **hourly bike rental demand** using time, weather, season, and working-day information.

## 📌 Features

* Generates a realistic bike rental dataset
* Cleans and preprocesses the data
* Performs Exploratory Data Analysis (EDA)
* Creates time-based and weather-based features
* Applies cyclical encoding for time features
* Trains and compares multiple regression models
* Evaluates models using MAE, RMSE, and R²
* Saves plots and model results automatically

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## 🤖 Machine Learning Models

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**

The model with the **lowest RMSE** is selected as the best-performing model.

## 📊 Features Used

* Hour
* Month
* Weekday
* Season
* Weather
* Temperature
* Humidity
* Windspeed
* Working Day
* Holiday
* Rush Hour
* Comfort Index

Cyclical encoding is applied to **hour, month, and weekday** to capture repeating time patterns.

## 📈 Evaluation Metrics

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* **R² Score** – Coefficient of Determination

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
├── bike_rental_prediction.py
└── README.md
```

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Bike_Rental_Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Set the Project Path

Update `BASE_DIR` in the Python script:

```python
BASE_DIR = r"C:\path\to\Bike_Rental_Prediction"
```

### 4. Run the Project

```bash
python bike_rental_prediction.py
```

The dataset, plots, and model results will automatically be saved in the `data/` and `plots/` folders.

## 📊 Output

The project generates:

* Bike rental dataset
* EDA visualizations
* Model comparison results
* Actual vs. predicted plot
* Feature importance plot
* `model_results.csv`

## 🎯 Objective

The objective is to analyze bike rental patterns and build a machine learning model that predicts **hourly bike rental demand** based on historical, time, and environmental factors.
