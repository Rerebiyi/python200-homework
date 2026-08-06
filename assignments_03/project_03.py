import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


# Task 2: Prepare Your Data

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

# Fit PCA only on the training data so the test data stays unseen.

# Fit PCA on the training data
pca = PCA()

pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(6, 4))

plt.plot(cumulative_variance)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")

plt.savefig("assignments_03/outputs/spambase_pca_variance.png")
plt.close()

n = np.argmax(cumulative_variance >= 0.90) + 1

print("Components for 90% variance:", n)

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

print("X_train_pca shape:", X_train_pca.shape)
print("X_test_pca shape:", X_test_pca.shape)