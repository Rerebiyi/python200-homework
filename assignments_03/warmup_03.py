import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---

# Question 1
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Question 2
scaler = StandardScaler()

# Fit the scaler using only the training data to prevent data leakage.
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Column means of X_train_scaled:")
print(X_train_scaled.mean(axis=0))

# --- KNN ---

# Question 1
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Question 2
knn_scaled = KNeighborsClassifier(n_neighbors=5)

knn_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("Scaled Accuracy:", accuracy_score(y_test, y_pred_scaled))

# Scaling made the accuracy a little lower for this dataset.

# Question 3
scores = cross_val_score(knn, X_train, y_train, cv=5)

print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())
print("Standard deviation:", scores.std())

# This is more trustworthy because it tests the model more than one time.

# Question 4
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print("k =", k, "Mean CV score:", scores.mean())

# I would choose k = 5 because it has the highest mean score.

# --- Classifier Evaluation ---

# Question 1
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot()

plt.tight_layout()
plt.savefig("assignments_03/outputs/knn_confusion_matrix.png")
plt.close()

# The model does not confuse any species because all predictions are correct.

# --- Decision Trees ---

# Question 1
tree = DecisionTreeClassifier(max_depth=3, random_state=42)

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, tree_pred))
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, tree_pred))

# The Decision Tree accuracy is lower than the KNN accuracy.

# Scaling should not change the result because Decision Trees do not use distance.


# --- Logistic Regression and Regularization ---

# Question 1
c_values = [0.01, 1.0, 100]

for c in c_values:
    model = OneVsRestClassifier(
        LogisticRegression(
            C=c,
            max_iter=1000,
            solver="liblinear"
        )
    )

    model.fit(X_train_scaled, y_train)

    coefficients = np.array([
        estimator.coef_[0]
        for estimator in model.estimators_
    ])

    total_coefficient_size = np.abs(coefficients).sum()

    print("C value:", c)
    print("Total coefficient size:", total_coefficient_size)

# As C increases, the total coefficient size increases.
# Regularization keeps the coefficients smaller.

# --- PCA ---

digits = load_digits()
X_digits = digits.data
y_digits = digits.target
images = digits.images

# Question1
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

fig, axes = plt.subplots(1, 10, figsize=(12, 2))

for digit in range(10):
    index = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[index], cmap="gray_r")
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")

plt.tight_layout()
plt.savefig("assignments_03/outputs/sample_digits.png")
plt.close()

# Question 2
pca = PCA()

pca.fit(X_digits)

scores = pca.transform(X_digits)

scatter = plt.scatter(
    scores[:, 0],
    scores[:, 1],
    c=y_digits,
    cmap="tab10",
    s=10
)

plt.colorbar(scatter, label="Digit")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Projection of Digits")

plt.savefig("assignments_03/outputs/pca_2d_projection.png")
plt.close()

# Yes, most of the same digits cluster together.

# Question 3
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")

plt.savefig("assignments_03/outputs/pca_variance_explained.png")
plt.close()

# About 13 components explain 80% of the variance.


# Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


component_values = [2, 5, 15, 40]

fig, axes = plt.subplots(5, 5, figsize=(8, 8))

# Original row
for i in range(5):
    axes[0, i].imshow(images[i], cmap="gray_r")
    axes[0, i].set_title(str(y_digits[i]))
    axes[0, i].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=10)

# Reconstruction rows
for row, n in enumerate(component_values, start=1):
    for i in range(5):
        reconstructed_image = reconstruct_digit(i, scores, pca, n)

        axes[row, i].imshow(reconstructed_image, cmap="gray_r")
        axes[row, i].axis("off")

    axes[row, 0].set_ylabel("n = " + str(n), fontsize=10)
plt.tight_layout()
plt.savefig("assignments_03/outputs/pca_reconstructions.png")
plt.close()

# The digits become clearly recognizable around n = 15.
# This matches where the variance curve starts to level off.
