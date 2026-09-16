# --- Step 1: Fetch the Data ---
import json
import joblib
import platform
import sklearn
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 40.59,
    "longitude": -74.46,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])

df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# Print dataset summary
print(df.head())
print()
df.info()
print()
print(df.describe())


# --- Step 2: Engineer Labels ---

# A good running day follows these weather conditions
df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7) &
    (df["temperature_2m_max"] <= 26) &
    (df["temperature_2m_min"] >= 0) &
    (df["precipitation_sum"] < 3.0) &
    (df["wind_speed_10m_max"] < 30)
).astype(int)

# Print class distribution
print()
print("Class Distribution:")
print(df["good_for_running"].value_counts())

# Print the fraction of good running days
good_fraction = df["good_for_running"].mean()
print("Fraction of good running days:", good_fraction)

# About 36% of the days are good for running.
# This seems reasonable because New Jersey has hot, cold, and rainy days.

# --- Step 3: Train and Tune ---

# Select features and label
X = df[
    [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
]

y = df["good_for_running"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build the pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression(max_iter=1000))
])

# C values to test
param_grid = {
    "log_reg__C": [0.01, 0.1, 1.0, 10.0, 100.0]
}

# Run GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc"
)

grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]

# Print results
print()
print("Best C:", grid_search.best_params_["log_reg__C"])
print("Best CV AUC:", grid_search.best_score_)
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

test_auc = roc_auc_score(y_test, y_probs)
print("Test AUC:", test_auc)

# Save ROC curve
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Weather Classifier ROC Curve")
plt.savefig("assignments_04/outputs/weather_roc.png")
plt.close()


# --- Step 4: Reflect on Evaluation ---

# The test AUC is about 0.70, so the model does a fair job separating the two classes.
# This is about what I expected from a simple weather model.
# The model has more false negatives because the recall for good running days is low.
# This means the app would miss some days that are actually good for running.
# I would rather the app recommend a few extra days than miss good running days.
# I would use a threshold lower than 0.5, such as 0.4, to catch more good running days.

# --- Step 5: Save the Model ---

# Save the best pipeline
joblib.dump(best_model, "assignments_04/models/weather_classifier.pkl")

# Model information
metadata = {
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
    "feature_names": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "best_hyperparameters": grid_search.best_params_,
    "test_auc": test_auc,
    "city": {
         "name": "Piscataway, NJ",
        "latitude": 40.59,
        "longitude": -74.46,
    },
    "label_thresholds": (
        "Max temperature 7-26 C, min temperature at least 0 C, "
        "precipitation under 3 mm, and wind speed under 30 km/h."
    ),
}

# Save the metadata
with open("assignments_04/models/weather_classifier_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

print("Model and metadata saved successfully.")