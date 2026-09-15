import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib

os.makedirs("assignments_04/outputs", exist_ok=True)
os.makedirs("assignments_04/models", exist_ok=True)
# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- ROC and AUC ---
# Q1

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Get probabilities
log_reg_probs = log_reg.predict_proba(X_test)[:, 1]

# Calculate AUC
log_reg_auc = roc_auc_score(y_test, log_reg_probs)
print("Logistic Regression AUC:", log_reg_auc)


# Scale the data for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Get probabilities
knn_probs = knn.predict_proba(X_test_scaled)[:, 1]

# Calculate AUC
knn_auc = roc_auc_score(y_test, knn_probs)
print("KNN AUC:", knn_auc)

# KNN has a higher AUC.
# KNN is better at telling the two classes apart.

# Question 2

# Calculate ROC curve values
log_fpr, log_tpr, _ = roc_curve(y_test, log_reg_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)

# Plot both ROC curves
plt.figure()

plt.plot(
    log_fpr,
    log_tpr,
    label=f"Logistic Regression (AUC = {log_reg_auc:.4f})"
)

plt.plot(
    knn_fpr,
    knn_tpr,
    label=f"KNN (AUC = {knn_auc:.4f})"
)

# Random classifier line
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

plt.savefig("assignments_04/outputs/roc_comparison.png")
plt.close()


# Find the FPR when TPR is closest to 0.80
log_index = np.argmin(np.abs(log_tpr - 0.80))
knn_index = np.argmin(np.abs(knn_tpr - 0.80))

print("Logistic Regression FPR at TPR 0.80:", log_fpr[log_index])
print("KNN FPR at TPR 0.80:", knn_fpr[knn_index])

# KNN has the lower FPR at TPR 0.80.
# KNN would have fewer false alarms while catching 80% of positives.


# Question 3

# Use Logistic Regression probabilities from Q1
y_probs_lr = log_reg_probs

# Get ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_probs_lr)

best_f1 = 0
best_threshold = 0
best_tpr = 0
best_fpr = 0

# Find the threshold with the highest F1 score
for i, threshold in enumerate(thresholds):
    y_pred = (y_probs_lr >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_tpr = tpr[i]
        best_fpr = fpr[i]

print("Best Threshold:", best_threshold)
print("TPR:", best_tpr)
print("FPR:", best_fpr)
print("Best F1 Score:", best_f1)

# The best threshold is lower than 0.5.
# I would use a lower threshold when catching more positives is more important.

# --- GridSearchCV ---
# Question 1

# Build the pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression(max_iter=1000))
])

# Set the C values to test
param_grid = {
    "log_reg__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
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
best_lr_pipe = grid_search.best_estimator_

# Get test probabilities
best_probs = best_lr_pipe.predict_proba(X_test)[:, 1]

# Calculate test AUC
best_test_auc = roc_auc_score(y_test, best_probs)

print("Best C:", grid_search.best_params_["log_reg__C"])
print("Best CV AUC:", grid_search.best_score_)
print("Best Test AUC:", best_test_auc)
# Grid search chose C=100.0 instead of the default C=1.0.
# The test AUC decreased by about 0.0003, so the performance was almost the same.

# Question 2

# Build the Decision Tree pipeline
tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("tree", DecisionTreeClassifier(random_state=42))
])

# Set the max_depth values to test
tree_param_grid = {
    "tree__max_depth": [2, 3, 5, 8, None]
}

# Run GridSearchCV
tree_grid_search = GridSearchCV(
    tree_pipeline,
    tree_param_grid,
    cv=5,
    scoring="roc_auc"
)

tree_grid_search.fit(X_train, y_train)

# Get the best model
best_tree = tree_grid_search.best_estimator_

# Get probabilities and test AUC
tree_probs = best_tree.predict_proba(X_test)[:, 1]
tree_test_auc = roc_auc_score(y_test, tree_probs)

print("Best max_depth:", tree_grid_search.best_params_["tree__max_depth"])
print("Best Decision Tree CV AUC:", tree_grid_search.best_score_)
print("Decision Tree Test AUC:", tree_test_auc)

# The Decision Tree has a higher AUC.
# I would choose the Decision Tree because it performed better.
# I would also check other scores like precision and recall.

# Question 3

# Get the grid search results
results = tree_grid_search.cv_results_

# Sort results from best to worst
sorted_results = sorted(
    zip(
        results["param_tree__max_depth"],
        results["mean_test_score"],
        results["std_test_score"]
    ),
    key=lambda x: x[1],
    reverse=True
)

# Print the mean and standard deviation for each max_depth
for max_depth, mean_auc, std_auc in sorted_results:
    print(
        "max_depth:", max_depth,
        "Mean CV AUC:", mean_auc,
        "Standard Deviation:", std_auc
    )
# max_depth 8 and None have similar mean AUC scores.
# max_depth 8 has a lower standard deviation.
# I would choose max_depth 8 because its performance is more consistent.

# --- joblib ---
# Question 1

# Get the best Logistic Regression pipeline
best_lr_pipe = grid_search.best_estimator_

# Save the pipeline
joblib.dump(best_lr_pipe, "assignments_04/models/warmup_model.pkl")

# Load the pipeline
loaded_clf = joblib.load("assignments_04/models/warmup_model.pkl")

# Compare predictions
original_preds = best_lr_pipe.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# Without the scaler, the data would not be scaled.
# This could make the predictions wrong.


# Question 2

# --- Simulated prediction script ---

# Load the saved model
loaded_model = joblib.load("assignments_04/models/warmup_model.pkl")

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

# Make predictions
new_preds = loaded_model.predict(new_samples)
new_probs = loaded_model.predict_proba(new_samples)

# Print each prediction
for i in range(len(new_samples)):
    print("Sample", i + 1)
    print("Predicted class:", new_preds[i])
    print("Probability:", new_probs[i][new_preds[i]])

# I expect the all-zero row to predict class 1.
# The model learned that this row is more likely to be class 1.