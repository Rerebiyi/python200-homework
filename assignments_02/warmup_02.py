import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

os.makedirs("assignments_02/outputs", exist_ok=True)

# --- scikit-learn API ---

# Question 1

# Create the data
years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# Create the model
model = LinearRegression()

# Fit the model
model.fit(years, salary)

# Make predictions
salary_4 = model.predict([[4]])
salary_8 = model.predict([[8]])

# Print the results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 4 years of experience:", salary_4[0])
print("Predicted salary for 8 years of experience:", salary_8[0])


# Question 2

# Create a 1D array
x = np.array([10, 20, 30, 40, 50])

# Print the original shape
print("Original shape:", x.shape)

# Reshape the array to 2D
x = x.reshape(-1, 1)

# Print the new shape
print("New shape:", x.shape)

# scikit-learn expects X to be 2D because each row represents one sample
# and each column represents one feature.


# Question 3

# Create a synthetic dataset with 3 clusters
X_clusters, _ = make_blobs(
    n_samples=120,
    centers=3,
    cluster_std=0.8,
    random_state=7
)

# Create the K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model to the data
kmeans.fit(X_clusters)

# Predict the cluster label for each point
labels = kmeans.predict(X_clusters)

# Print the cluster centers
print("Cluster centers:")
print(kmeans.cluster_centers_)

# Print the number of points in each cluster
print("Number of points in each cluster:")
print(np.bincount(labels))

# Create the scatter plot
plt.figure()

plt.scatter(
    X_clusters[:, 0],
    X_clusters[:, 1],
    c=labels
)

# Plot the cluster centers as black X's
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color="black",
    marker="x",
    s=200
)

# Add a title and axis labels
plt.title("K-Means Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Save the figure
plt.savefig("assignments_02/outputs/kmeans_clusters.png")

# Close the figure
plt.close()


# --- Linear Regression ---

# Create the synthetic medical costs dataset
np.random.seed(42)

num_patients = 100

age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)

cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)


# Question 1

# Create a scatter plot of age versus medical cost
plt.figure()

plt.scatter(age, cost, c=smoker, cmap="coolwarm")

# Add a title and axis labels
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost")

# Save the figure
plt.savefig("assignments_02/outputs/cost_vs_age.png")

# Close the figure
plt.close()

# The plot shows two distinct groups of points.
# This suggests that smoker status has a strong effect on medical cost,
# with smokers generally having higher medical costs than non-smokers.


# Question 2

# Reshape age into a 2D array
X = age.reshape(-1, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    cost,
    test_size=0.2,
    random_state=42
)

# Print the shapes of each array
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Question 3

# Create the linear regression model
model = LinearRegression()

# Fit the model using the training data
model.fit(X_train, y_train)

# Print the slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict medical costs for the test data
y_pred = model.predict(X_test)

# Calculate the RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

# Print the evaluation metrics
print("RMSE:", rmse)
print("R² on the test set:", model.score(X_test, y_test))

# The slope represents the estimated increase in medical cost
# for each additional year of age.


# Question 4

# Create a feature matrix with age and smoker status
X_full = np.column_stack([age, smoker])

# Split the data into training and test sets
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    cost,
    test_size=0.2,
    random_state=42
)

# Create and fit the linear regression model
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

# Print the R² score on the test set
print("R² on the test set (age only):", model.score(X_test, y_test))
print("R² on the test set (age and smoker):", model_full.score(X_test_full, y_test_full))

# Print the coefficients
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# Adding the smoker feature improves the model because the R² score increases.
# The smoker coefficient represents the predicted increase in medical cost
# for smokers compared with non-smokers of the same age.


# Question 5

# Predict the medical costs for the test set
y_pred_full = model_full.predict(X_test_full)

# Create the scatter plot
plt.figure()

plt.scatter(y_pred_full, y_test_full)

# Create a diagonal reference line
min_value = min(y_test_full.min(), y_pred_full.min())
max_value = max(y_test_full.max(), y_pred_full.max())

plt.plot([min_value, max_value], [min_value, max_value], color="red")

# Add a title and axis labels
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Medical Cost")
plt.ylabel("Actual Medical Cost")

# Save the figure
plt.savefig("assignments_02/outputs/predicted_vs_actual_cost.png")

# Close the figure
plt.close()

# A point above the diagonal means the actual medical cost is higher
# than the model predicted.
# A point below the diagonal means the model predicted a higher
# medical cost than the actual value.