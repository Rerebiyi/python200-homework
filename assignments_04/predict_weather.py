# --- Task 1: Load and Verify ---

import json
import joblib

# Load the saved model
model = joblib.load("assignments_04/models/weather_classifier.pkl")

# Load the metadata
with open("assignments_04/models/weather_classifier_metadata.json", "r") as file:
    metadata = json.load(file)

# Print key metadata
print("City:", metadata["city"])
print("Features:", metadata["feature_names"])
print("Test AUC:", metadata["test_auc"])


# --- Task 2: Predict on New Data ---

import pandas as pd

# Create sample weather days
new_days = pd.DataFrame(
    [
        [20, 10, 0.0, 12],   # Good
        [24, 15, 1.0, 18],   # Good
        [32, 22, 0.0, 10],   # Too hot
        [5, -2, 0.0, 8],     # Too cold
        [26, 5, 2.9, 29],    # Borderline
    ],
    columns=metadata["feature_names"]
)

# Make predictions
predictions = model.predict(new_days)
probabilities = model.predict_proba(new_days)[:, 1]

# Print results
for i in range(len(new_days)):
    print()
    print("Day", i + 1)
    print("Temperature max:", new_days.iloc[i]["temperature_2m_max"])
    print("Temperature min:", new_days.iloc[i]["temperature_2m_min"])
    print("Precipitation:", new_days.iloc[i]["precipitation_sum"])
    print("Wind speed:", new_days.iloc[i]["wind_speed_10m_max"])

    if predictions[i] == 1:
        print("Prediction: good")
    else:
        print("Prediction: skip")

    print("Confidence:", probabilities[i])

    # --- Task 3: Reflect ---

# My borderline day had a probability of about 0.05, so the model was confident it was not good.
# If the probability was 0.52, I would consider the model uncertain because it is close to 0.5.

# If the training script was not run first, the saved model would not exist.
# I would add an error message that tells the user to run the training script first.

# For daily predictions, the script would need to get tomorrow's weather data.
# It would then use that weather data with the saved model to make a prediction.
