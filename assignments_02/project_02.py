# This CSV file uses semicolons instead of commas,
# so pd.read_csv() needs sep=";".

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("assignments_02/outputs", exist_ok=True)

# Task 1: Load and Explore      

# Load the dataset
df = pd.read_csv(
    "assignments_02/student_performance_math.csv",
    sep=";"
)

# Print the shape
print("Shape:")
print(df.shape)

# Print the first five rows
print("\nFirst five rows:")
print(df.head())

# Print the data types
print("\nData types:")
print(df.dtypes)

# Create a histogram of final grades
plt.figure()
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.savefig("assignments_02/outputs/g3_distribution.png")
plt.close()


# Task 2: Preprocess the Data

# Students with G3 = 0 did not take the final exam.
# Keeping them could confuse the model because zero was not their real grade.
df_clean = df[df["G3"] != 0].copy()

# Print the shapes before and after filtering
print("\nShape before filtering:")
print(df.shape)

print("\nShape after filtering:")
print(df_clean.shape)

print("\nRows removed:")
print(df.shape[0] - df_clean.shape[0])

# Convert yes/no columns to 1/0
yes_no_columns = [
    "schoolsup",
    "internet",
    "higher",
    "activities"
]

for column in yes_no_columns:
    df_clean[column] = df_clean[column].map(
        {"no": 0, "yes": 1}
    )

# Convert sex to 0/1
df_clean["sex"] = df_clean["sex"].map(
    {"F": 0, "M": 1}
)

# Calculate correlations before and after filtering
original_correlation = df["absences"].corr(df["G3"])
filtered_correlation = df_clean["absences"].corr(
    df_clean["G3"]
)

print("\nCorrelation between absences and G3 before filtering:")
print(original_correlation)

print("\nCorrelation between absences and G3 after filtering:")
print(filtered_correlation)

# Some students with G3 = 0 had low absences.
# This made absences look less connected to final grades.
# Removing those rows shows the relationship more clearly.


# Task 3: Exploratory Data Analysis

# Find the correlation between each numeric feature and G3
numeric_features = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "absences",
    "freetime",
    "goout",
    "Walc"
]

correlations = (
    df_clean[numeric_features + ["G3"]]
    .corr()["G3"]
    .drop("G3")
    .sort_values()
)

print("\nCorrelations with G3:")
print(correlations)

print("\nStrongest relationship with G3:")
print(
    "Ignoring G1 and G2, failures has the strongest "
    "relationship with G3."
)

print("\nSurprising result:")
print("Study time has only a weak positive correlation with G3.")

# Students with more past failures usually have lower final grades.

plt.figure()
plt.scatter(df_clean["failures"], df_clean["G3"])
plt.title("Failures vs Final Grade")
plt.xlabel("Number of Past Failures")
plt.ylabel("Final Grade (G3)")
plt.savefig("assignments_02/outputs/failures_vs_g3.png")
plt.close()

# Students with more absences often have lower grades,
# but the pattern is not very strong.

plt.figure()
plt.scatter(df_clean["absences"], df_clean["G3"])
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.savefig("assignments_02/outputs/absences_vs_g3.png")
plt.close()


# Task 4: Baseline Model

# Use failures as the only feature
X_baseline = df_clean[["failures"]]
y_baseline = df_clean["G3"]

# Split the data
X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = (
    train_test_split(
        X_baseline,
        y_baseline,
        test_size=0.2,
        random_state=42
    )
)

# Train the baseline model
baseline_model = LinearRegression()
baseline_model.fit(
    X_train_baseline,
    y_train_baseline
)

# Make predictions
baseline_predictions = baseline_model.predict(
    X_test_baseline
)

# Calculate evaluation metrics
baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test_baseline,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    y_test_baseline,
    baseline_predictions
)

print("\nBaseline Model Results")
print("Slope:", baseline_model.coef_[0])
print("RMSE:", baseline_rmse)
print("R²:", baseline_r2)

# More past failures usually mean lower predicted grades.
# The model is usually off by a few grade points.
# Failures alone are not enough to predict final grades well.


# Task 5: Build the Full Model

feature_cols = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "absences",
    "freetime",
    "goout",
    "Walc",
    "schoolsup",
    "internet",
    "higher",
    "activities",
    "sex"
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the full model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate evaluation metrics
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

print("\nFull Model Results")
print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("Test RMSE:", test_rmse)

print("\nImprovement over Baseline")
print("Baseline Test R²:", baseline_r2)
print("Full Model Test R²:", test_r2)
print("R² Improvement:", test_r2 - baseline_r2)

print("\nTrain vs Test")


if train_r2 - test_r2 < 0.10:
    print("The train and test R² values are close.")
    print("The model works similarly on new data.")
else:
    print("The train R² is higher than the test R².")
    print("The model may not work as well on new data.")

print("\nFeature Coefficients")

for name, coefficient in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coefficient:+.3f}")

# Some coefficients may be surprising because the features can affect
# each other when they are used together in one model.

# I would keep features that have larger coefficients and seem useful,
# such as failures, studytime, absences, and higher.
# I would consider removing features with coefficients close to zero.
# I would also be careful about using sex because it is a personal feature.

# Task 6: Evaluate and Summarize

# Create a predicted vs actual plot
plt.figure()
plt.scatter(test_predictions, y_test)

# Set the range of the diagonal line
line_min = min(test_predictions.min(), y_test.min())
line_max = max(test_predictions.max(), y_test.max())

# Add a diagonal reference line
plt.plot(
    [line_min, line_max],
    [line_min, line_max]
)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")

plt.savefig(
    "assignments_02/outputs/predicted_vs_actual_g3.png"
)
plt.close()

# Points close to the line are good predictions.
# Points above the line mean the prediction was too low.
# Points below the line mean the prediction was too high.
# The errors appear across the grade levels instead of in only one area.

print("\nSummary")
print("The filtered dataset has", df_clean.shape[0], "students.")
print("The test set has", len(y_test), "students.")

print(
    "The model is usually off by about",
    round(test_rmse, 2),
    "grade points on the 0 to 20 scale."
)
# The model is usually off by about 2.66 grade points.
# It explains about 26% of the differences in final grades.
print(
    "The model explains about",
    round(test_r2 * 100, 1),
    "% of the variation in final grades."
)

# Find the largest positive and negative coefficients
largest_positive = np.argmax(model.coef_)
largest_negative = np.argmin(model.coef_)

positive_feature = feature_cols[largest_positive]
positive_coefficient = model.coef_[largest_positive]

negative_feature = feature_cols[largest_negative]
negative_coefficient = model.coef_[largest_negative]

print("\nLargest Positive Coefficient:")
print(positive_feature, positive_coefficient)

print(
    positive_feature,
    "has the largest positive effect on the predicted grade."
)

print("\nLargest Negative Coefficient:")
print(negative_feature, negative_coefficient)

print(
    negative_feature,
    "has the largest negative effect on the predicted grade."
)

print("\nOne Surprising Result:")
print("Study time had only a small relationship with final grades.")


# Neglected Feature: The Power of G1

feature_cols_g1 = feature_cols + ["G1"]

X_g1 = df_clean[feature_cols_g1].values
y_g1 = df_clean["G3"].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1,
    y_g1,
    test_size=0.2,
    random_state=42
)

# Train the model with G1
model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

# Make predictions
predictions_g1 = model_g1.predict(X_test_g1)

# Calculate the new test R²
g1_r2 = r2_score(y_test_g1, predictions_g1)

print("\nModel with G1")
print("Test R²:", g1_r2)

# A high R² does not mean G1 causes G3.
# G1 and G3 are grades from the same class, so they are closely related.
# G1 can help find students who may struggle later.
# To help students earlier, schools would need to use
# past failures, absences, study habits, and other early information.