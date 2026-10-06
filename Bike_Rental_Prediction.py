"""
Bike Rental Demand Prediction — Full Pipeline
------------------------------------------------
Data generation -> Preprocessing -> EDA -> Feature Engineering -> Modeling

HOW TO USE:
1. Set BASE_DIR below to your project folder.
2. Run this script top to bottom in VS Code / Jupyter.
   All data, plots, and results will be saved inside BASE_DIR.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =====================================================================
# 0. SET YOUR PROJECT FOLDER HERE (raw string — note the r before quotes)
# =====================================================================
BASE_DIR = r"C:\Users\SangeethaDola\OneDrive\AI_Batch_2_Use_Case\Bike_Rental_Prediction"

DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "bike_rentals_dataset.csv")
RESULTS_PATH = os.path.join(DATA_DIR, "model_results.csv")

print("Data will be saved to:", CSV_PATH)
print("Plots will be saved to:", PLOTS_DIR)


# =====================================================================
# 1. GENERATE DATASET (~1000 rows, realistic hourly rental patterns)
#    Skip this whole section if you already have bike_rentals_dataset.csv
#    — just move that file into DATA_DIR and go to Section 2.
# =====================================================================
N_DAYS = 42
HOURS_PER_DAY = 24
N_ROWS_TARGET = 1000

start_date = pd.Timestamp("2024-01-01")
dates = pd.date_range(start_date, periods=N_DAYS, freq="D")

records = []
for day in dates:
    season = (
        1 if day.month in [12, 1, 2] else
        2 if day.month in [3, 4, 5] else
        3 if day.month in [6, 7, 8] else
        4
    )
    is_weekend = day.dayofweek >= 5
    is_holiday = 1 if (day.month == 1 and day.day == 1) else 0
    workingday = 0 if (is_weekend or is_holiday) else 1

    season_base_temp = {1: 6, 2: 16, 3: 28, 4: 18}[season]
    day_temp_shift = np.random.normal(0, 3)

    for hour in range(HOURS_PER_DAY):
        temp = season_base_temp + day_temp_shift + 6 * np.sin((hour - 6) / 24 * 2 * np.pi)
        temp += np.random.normal(0, 1.2)
        atemp = temp + np.random.normal(1, 0.8)

        humidity = np.clip(60 - 0.6 * temp + np.random.normal(0, 8), 20, 100)
        windspeed = np.clip(np.random.gamma(2, 5), 0, 45)

        weather_roll = np.random.random()
        if weather_roll < 0.60:
            weathersit = 1
        elif weather_roll < 0.85:
            weathersit = 2
        elif weather_roll < 0.97:
            weathersit = 3
        else:
            weathersit = 4

        if workingday:
            base_demand = (
                80 * np.exp(-((hour - 8) ** 2) / 4) +
                110 * np.exp(-((hour - 17.5) ** 2) / 6) +
                15
            )
        else:
            base_demand = (
                90 * np.exp(-((hour - 13) ** 2) / 20) +
                20
            )

        temp_effect = np.clip(1 + 0.035 * (temp - 20), 0.3, 1.9)
        weather_effect = {1: 1.0, 2: 0.85, 3: 0.5, 4: 0.15}[weathersit]
        humidity_effect = np.clip(1.15 - 0.005 * humidity, 0.6, 1.15)
        wind_effect = np.clip(1.05 - 0.01 * windspeed, 0.6, 1.05)
        season_effect = {1: 0.75, 2: 1.0, 3: 1.15, 4: 0.95}[season]

        expected_count = max(
            base_demand * temp_effect * weather_effect *
            humidity_effect * wind_effect * season_effect,
            1
        )
        count = np.random.poisson(expected_count)

        registered_share = 0.85 if workingday and hour in [7, 8, 9, 16, 17, 18, 19] else 0.55
        registered = int(count * np.clip(registered_share + np.random.normal(0, 0.05), 0, 1))
        casual = max(count - registered, 0)

        records.append({
            "dteday": day.strftime("%Y-%m-%d"),
            "season": season, "yr": day.year - start_date.year, "mnth": day.month,
            "hr": hour, "holiday": is_holiday, "weekday": day.dayofweek,
            "workingday": workingday, "weathersit": weathersit,
            "temp": round(temp, 2), "atemp": round(atemp, 2),
            "hum": round(humidity, 1), "windspeed": round(windspeed, 1),
            "casual": casual, "registered": registered, "cnt": count,
        })

df = pd.DataFrame(records)
if len(df) > N_ROWS_TARGET:
    df = df.sample(n=N_ROWS_TARGET, random_state=42).sort_values(["dteday", "hr"]).reset_index(drop=True)
df.insert(0, "instant", range(1, len(df) + 1))

df.to_csv(CSV_PATH, index=False)
print(f"\nSaved {len(df)} rows to {CSV_PATH}")


# =====================================================================
# 2. LOAD + PREPROCESS
# =====================================================================
df = pd.read_csv(CSV_PATH, parse_dates=["dteday"])

print("\nShape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().dropna()
cat_cols = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
for c in cat_cols:
    df[c] = df[c].astype(int)

print("\nSummary statistics:\n",
      df[["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]].describe())


# =====================================================================
# 3. EDA
# =====================================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["cnt"], bins=30, kde=True, ax=ax[0], color="steelblue")
ax[0].set_title("Distribution of Hourly Rental Count")
sns.histplot(np.log1p(df["cnt"]), bins=30, kde=True, ax=ax[1], color="darkorange")
ax[1].set_title("Distribution of log(1 + Rental Count)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_target_distribution.png"))
plt.close()

hourly = df.groupby(["hr", "workingday"])["cnt"].mean().reset_index()
hourly["Working Day"] = hourly["workingday"].map({0: "No", 1: "Yes"})
plt.figure(figsize=(9, 4.5))
sns.lineplot(data=hourly, x="hr", y="cnt", hue="Working Day", marker="o",
             hue_order=["No", "Yes"], palette={"No": "tomato", "Yes": "steelblue"})
plt.title("Average Rentals by Hour of Day (Working Day vs Not)")
plt.xlabel("Hour of Day"); plt.ylabel("Average Rental Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_hourly_pattern.png"))
plt.close()

season_labels = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
df["season_label"] = df["season"].map(season_labels)
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df, x="season_label", y="cnt", hue="season_label", legend=False,
            order=["Winter", "Spring", "Summer", "Fall"], palette="viridis")
plt.title("Rental Count Distribution by Season")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_season_boxplot.png"))
plt.close()

weather_labels = {1: "Clear", 2: "Mist/Cloudy", 3: "Light Rain/Snow", 4: "Heavy Rain/Snow"}
df["weather_label"] = df["weathersit"].map(weather_labels)
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df, x="weather_label", y="cnt", hue="weather_label", legend=False,
            order=["Clear", "Mist/Cloudy", "Light Rain/Snow", "Heavy Rain/Snow"], palette="coolwarm")
plt.title("Rental Count by Weather Situation")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_weather_boxplot.png"))
plt.close()

plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df, x="temp", y="cnt", hue="season_label", alpha=0.6, s=25, palette="viridis")
plt.title("Rental Count vs Temperature")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_temp_scatter.png"))
plt.close()

num_cols = ["temp", "atemp", "hum", "windspeed", "hr", "cnt"]
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_correlation_heatmap.png"))
plt.close()

print("\nEDA plots saved to", PLOTS_DIR)


# =====================================================================
# 4. FEATURE ENGINEERING
# =====================================================================
# Drop the plot-only label columns added during EDA (their names start
# with "season_"/"weather_", which would otherwise get swept into the
# feature list below and break the model with text values).
df = df.drop(columns=["season_label", "weather_label"])

df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
df["is_rush_hour"] = df["hr"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
df["comfort_index"] = df["temp"] - 0.1 * df["hum"] - 0.1 * df["windspeed"]

df = pd.get_dummies(df, columns=["season", "weathersit"], prefix=["season", "weather"], drop_first=True)

feature_cols = [
    "yr", "mnth_sin", "mnth_cos", "hr_sin", "hr_cos", "weekday_sin", "weekday_cos",
    "holiday", "workingday", "is_rush_hour",
    "temp", "atemp", "hum", "windspeed", "comfort_index",
] + [c for c in df.columns if c.startswith("season_") or c.startswith("weather_")]

X = df[feature_cols]
y = df["cnt"]
print(f"\nFeature matrix shape: {X.shape}")


# =====================================================================
# 5. MODEL TRAINING & COMPARISON
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2,
                                            random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                    random_state=RANDOM_STATE),
}

results = []
predictions = {}
for name, model in models.items():
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    preds = np.clip(preds, 0, None)
    predictions[name] = preds

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

results_df = pd.DataFrame(results).sort_values("RMSE")
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model (lowest RMSE): {best_model_name}")

# --- plots ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
    sns.barplot(data=results_df, x="Model", y=metric, hue="Model", legend=False, ax=ax, palette="mako")
    ax.set_title(metric); ax.set_xlabel(""); ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_model_comparison.png"))
plt.close()

best_preds = predictions[best_model_name]
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.5, color="teal", s=25)
lims = [0, max(y_test.max(), best_preds.max()) + 5]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Rental Count"); plt.ylabel("Predicted Rental Count")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_actual_vs_predicted.png"))
plt.close()

plt.figure(figsize=(7, 6))
if best_model_name == "Linear Regression":
    importances = pd.Series(models[best_model_name].coef_, index=feature_cols)
else:
    importances = pd.Series(models[best_model_name].feature_importances_, index=feature_cols)
importances = importances.sort_values(key=abs, ascending=True).tail(15)
importances.plot(kind="barh", color="slateblue")
plt.title(f"Top Feature Importances ({best_model_name})")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_feature_importance.png"))
plt.close()

results_df.to_csv(RESULTS_PATH, index=False)
print(f"\nSaved model comparison to {RESULTS_PATH}")
print(f"All plots saved to {PLOTS_DIR}")
