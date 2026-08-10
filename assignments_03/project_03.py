import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

#Task 1: Load the dataset and explore it 

# Load the feature names
column_names = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
    "spam_label"
]

# Load the dataset
df = pd.read_csv(
    "assignments_03/data/spambase/spambase.data",
    header=None,
    names=column_names
)

print("Number of emails:", len(df))

print("\nClass counts:")
print(df["spam_label"].value_counts())

print("\nClass percentages:")
print(df["spam_label"].value_counts(normalize=True) * 100)

# There are 4,601 emails in the dataset.

# The classes are fairly balanced.
# About 61% are ham and 39% are spam.

# The classes are a little different in size.
# Accuracy is helpful, but it is not enough by itself.

# Boxplot 1:

plt.figure(figsize=(6, 4))

df.boxplot(column="word_freq_free", by="spam_label")

plt.title("Word Frequency: free")
plt.suptitle("")
plt.xlabel("Spam Label")
plt.ylabel("Frequency")

plt.savefig("assignments_03/outputs/word_freq_free_boxplot.png")
plt.close()

# Boxplot 2:

plt.figure(figsize=(6, 4))

df.boxplot(column="char_freq_!", by="spam_label")

plt.title("Character Frequency: !")
plt.suptitle("")
plt.xlabel("Spam Label")
plt.ylabel("Frequency")

plt.savefig("assignments_03/outputs/char_freq_exclamation_boxplot.png")
plt.close()


# Boxplot 3:
plt.figure(figsize=(6, 4))

df.boxplot(column="capital_run_length_total", by="spam_label")

plt.title("Capital Run Length Total")
plt.suptitle("")
plt.xlabel("Spam Label")
plt.ylabel("Value")

plt.savefig("assignments_03/outputs/capital_run_length_total_boxplot.png")
plt.close()

# Spam emails usually use the word "free" more than ham emails.

# Spam emails usually use more "!" characters than ham emails.

# Spam emails usually have more capital letters than ham emails.
# The differences between spam and ham are clear, but there is still some overlap.

# Many values are 0 because most emails do not use every word or character.

# The scales are different because word frequencies are small values,
# while capital letter totals can be much larger.

# KNN and Logistic Regression work better when the features are on a similar scale.


# Task 2: Prepare Your Data

# I use an 80/20 split and stratify so both sets keep a similar class balance.
# Split the data into features and target
X = df.drop("spam_label", axis=1)
y = df["spam_label"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale the data
scaler = StandardScaler()

# Fit the scaler only on the training data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# I fit the scaler only on the training data so the test data stays unseen.

# Fit PCA only on the training data so the test data stays unseen.

# Fit PCA on the training data
pca = PCA(svd_solver="full")

pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(6, 4))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")

plt.savefig("assignments_03/outputs/spambase_pca_variance.png")
plt.close()

n = np.argmax(cumulative_variance >= 0.90) + 1

print("Components for 90% variance:", n)

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

# I think PCA may make the data smaller, but it may not improve accuracy.

print("X_train_pca shape:", X_train_pca.shape)
print("X_test_pca shape:", X_test_pca.shape)



# Task 3: A Classifier Comparison

# KNN on unscaled data
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)

knn_unscaled_pred = knn_unscaled.predict(X_test)

print("\nKNN Unscaled Accuracy:")
print(accuracy_score(y_test, knn_unscaled_pred))

print("KNN Unscaled Classification Report:")
print(classification_report(y_test, knn_unscaled_pred))


# KNN on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

knn_scaled_pred = knn_scaled.predict(X_test_scaled)

print("\nKNN Scaled Accuracy:")
print(accuracy_score(y_test, knn_scaled_pred))

print("KNN Scaled Classification Report:")
print(classification_report(y_test, knn_scaled_pred))


# KNN on PCA data
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)

knn_pca_pred = knn_pca.predict(X_test_pca)

print("\nKNN PCA Accuracy:")
print(accuracy_score(y_test, knn_pca_pred))

print("KNN PCA Classification Report:")
print(classification_report(y_test, knn_pca_pred))

# Scaling improved KNN a lot compared to the unscaled data.

# Scaled KNN performed slightly better than PCA KNN.

# Scaled KNN accuracy was about 0.908, while PCA KNN was about 0.907.

# I would choose scaled KNN because it had the slightly higher test accuracy.

# This matches my Task 2 guess that PCA might reduce the data without improving accuracy.

# Decision Tree

depths = [3, 5, 10, None]

for depth in depths:
    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_accuracy = tree.score(X_train, y_train)
    test_accuracy = tree.score(X_test, y_test)

    print("\nMax Depth:", depth)
    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

# Final Decision Tree model

final_tree = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

final_tree.fit(X_train, y_train)

final_tree_pred = final_tree.predict(X_test)

print("\nDecision Tree Accuracy:")
print(accuracy_score(y_test, final_tree_pred))

print("Decision Tree Classification Report:")
print(classification_report(y_test, final_tree_pred))

# The Decision Tree had about the same accuracy as KNN.

# Scaling does not make much difference for Decision Trees
# because they do not use distances between data points.
# As the tree gets deeper, the training accuracy gets higher.

# The test accuracy improves only a little.

# This shows that deeper trees can overfit the training data.

# I would choose max_depth = 10 for production.

# It gives strong test accuracy without almost memorizing the training data.

# The unlimited tree has almost perfect training accuracy, which shows more overfitting.
# Random Forest

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Logistic Regression on scaled data

logistic_scaled = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

logistic_scaled.fit(X_train_scaled, y_train)

logistic_scaled_pred = logistic_scaled.predict(X_test_scaled)

print("\nLogistic Regression Scaled Accuracy:")
print(accuracy_score(y_test, logistic_scaled_pred))

print("Logistic Regression Scaled Classification Report:")
print(classification_report(y_test, logistic_scaled_pred))

# Logistic Regression on PCA data

logistic_pca = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

logistic_pca.fit(X_train_pca, y_train)

logistic_pca_pred = logistic_pca.predict(X_test_pca)

print("\nLogistic Regression PCA Accuracy:")
print(accuracy_score(y_test, logistic_pca_pred))

print("Logistic Regression PCA Classification Report:")
print(classification_report(y_test, logistic_pca_pred))

# Decision Tree feature importance

tree_importances = pd.Series(
    final_tree.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Decision Tree Features:")
print(tree_importances.head(10))

# Random Forest feature importance

rf_importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Random Forest Features:")
print(rf_importances.head(10))

top_rf_features = rf_importances.head(10)

plt.figure(figsize=(8, 5))

plt.barh(
    top_rf_features.index[::-1],
    top_rf_features.values[::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Features")

plt.tight_layout()
plt.savefig("assignments_03/outputs/feature_importances.png")
plt.close()

# Both models agree that features like "!", "$", and "free" are important.

# The Random Forest and Decision Tree rank them a little differently, but many of the same features appear in both.

# These results make sense because spam emails often contain these words and characters.

# Random Forest had the highest overall test accuracy.

# Scaled KNN performed slightly better than PCA KNN.

# Logistic Regression also performed better with scaled data than PCA data.

# PCA did not improve test accuracy for either model.

# This matches my Task 2 guess that PCA might reduce the data without improving accuracy.

# Accuracy is not the only thing I would focus on.

# For a spam filter, I would rather reduce false positives.

# It is better to let a spam email through than send a real email to the spam folder.

# Confusion Matrix for the best model

cm = confusion_matrix(y_test, rf_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm
).plot()

plt.savefig(
    "assignments_03/outputs/best_model_confusion_matrix.png"
)
plt.close()

# The Random Forest makes more false negatives than false positives.

# This means more spam emails get through than real emails are marked as spam.



# Task 4: Cross-Validation

print("\nCross-Validation Results")

# KNN (unscaled)
knn_unscaled_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=5),
    X_train,
    y_train,
    cv=5
)

print("\nKNN Unscaled")
print("Mean:", knn_unscaled_scores.mean())
print("Standard Deviation:", knn_unscaled_scores.std())


# KNN (scaled)
knn_scaled_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=5),
    X_train_scaled,
    y_train,
    cv=5
)

print("\nKNN Scaled")
print("Mean:", knn_scaled_scores.mean())
print("Standard Deviation:", knn_scaled_scores.std())


# KNN (PCA)
knn_pca_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=5),
    X_train_pca,
    y_train,
    cv=5
)

print("\nKNN PCA")
print("Mean:", knn_pca_scores.mean())
print("Standard Deviation:", knn_pca_scores.std())


# Decision Tree
decision_tree_scores = cross_val_score(
    DecisionTreeClassifier(max_depth=10, random_state=42),
    X_train,
    y_train,
    cv=5
)

print("\nDecision Tree")
print("Mean:", decision_tree_scores.mean())
print("Standard Deviation:", decision_tree_scores.std())


# Random Forest
random_forest_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train,
    y_train,
    cv=5
)

print("\nRandom Forest")
print("Mean:", random_forest_scores.mean())
print("Standard Deviation:", random_forest_scores.std())


# Logistic Regression (scaled)
logistic_scaled_scores = cross_val_score(
    LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="liblinear"
    ),
    X_train_scaled,
    y_train,
    cv=5
)

print("\nLogistic Regression Scaled")
print("Mean:", logistic_scaled_scores.mean())
print("Standard Deviation:", logistic_scaled_scores.std())


# Logistic Regression (PCA)
logistic_pca_scores = cross_val_score(
    LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="liblinear"
    ),
    X_train_pca,
    y_train,
    cv=5
)

print("\nLogistic Regression PCA")
print("Mean:", logistic_pca_scores.mean())
print("Standard Deviation:", logistic_pca_scores.std())

# Random Forest had the highest average accuracy.

# Logistic Regression with PCA had the lowest standard deviation,
# so it was the most stable.

# The ranking was very similar to the train and test results.
# Random Forest was still the best model.

# Random Forest also had a lower standard deviation than the Decision Tree.

# Task 5: Building a Prediction Pipeline

# Random Forest pipeline
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)

rf_pipeline_pred = rf_pipeline.predict(X_test)

print("\nRandom Forest Pipeline Classification Report:")
print(classification_report(y_test, rf_pipeline_pred))


# Logistic Regression pipeline
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="liblinear"
    ))
])

logistic_pipeline.fit(X_train, y_train)

logistic_pipeline_pred = logistic_pipeline.predict(X_test)

print("\nLogistic Regression Pipeline Classification Report:")
print(classification_report(y_test, logistic_pipeline_pred))


# The pipelines do not have the same structure.
# Logistic Regression needs scaling, but Random Forest does not.

# PCA is not included because it did not improve Logistic Regression.

# Pipelines keep the model steps together.
# This makes the model easier to use and share with someone else.