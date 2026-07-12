# --- Pandas ---

import pandas as pd

# Pandas Q1

data = {
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade": [85, 72, 90, 68, 95],
    "city": ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)

print("First 3 rows:")
print(df.head(3))

print(f"\nShape: {df.shape}")

print("\nData types:")
print(df.dtypes)

# Pandas Q2

filtered_df = df[(df["passed"] == True) & (df["grade"] > 80)]

print("\nStudents who passed and have a grade above 80:")
print(filtered_df)

# Pandas Q3

df["grade_curved"] = df["grade"] + 5

print("\nDataFrame with curved grades:")
print(df)

# Pandas Q4

df["name_upper"] = df["name"].str.upper()

print("\nNames in uppercase:")
print(df[["name", "name_upper"]])

# Pandas Q5

city_mean = df.groupby("city")["grade"].mean()

print("\nMean grade by city:")
print(city_mean)

# Pandas Q6

df["city"] = df["city"].replace("Austin", "Houston")

print("\nUpdated city names:")
print(df[["name", "city"]])

# Pandas Q7

sorted_df = df.sort_values(by="grade", ascending=False)

print("\nTop 3 students by grade:")
print(sorted_df.head(3))

# --- NumPy ---

import numpy as np

# NumPy Q1

arr = np.array([10, 20, 30, 40, 50])

print("\nNumPy array:")
print(arr)

print(f"Shape: {arr.shape}")
print(f"Data type: {arr.dtype}")
print(f"Number of dimensions: {arr.ndim}")

# NumPy Q2

arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("\n2D NumPy array:")
print(arr)

print(f"Shape: {arr.shape}")
print(f"Size: {arr.size}")

# NumPy Q3

top_left = arr[:2, :2]

print("\nTop-left 2x2 block:")
print(top_left)

# NumPy Q4

zeros_array = np.zeros((3, 4))

ones_array = np.ones((2, 5))

print("\n3x4 array of zeros:")
print(zeros_array)

print("\n2x5 array of ones:")
print(ones_array)

# NumPy Q5

arr = np.arange(0, 50, 5)

print("\nArray created with np.arange:")
print(arr)

print(f"Shape: {arr.shape}")
print(f"Mean: {arr.mean()}")
print(f"Sum: {arr.sum()}")
print(f"Standard deviation: {arr.std()}")

# NumPy Q6

random_array = np.random.normal(0, 1, 200)

print("\nRandom normal distribution:")
print(f"Mean: {random_array.mean()}")
print(f"Standard deviation: {random_array.std()}")

# --- Matplotlib ---

import matplotlib.pyplot as plt

# Matplotlib Q1

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)

plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Matplotlib Q2

subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

plt.figure()

plt.bar(subjects, scores)

plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")

plt.show()

# Matplotlib Q3

x1 = [1, 2, 3, 4, 5]
y1 = [2, 4, 5, 4, 5]

x2 = [1, 2, 3, 4, 5]
y2 = [5, 4, 3, 2, 1]

plt.figure()

plt.scatter(x1, y1, color="blue", label="Dataset 1")
plt.scatter(x2, y2, color="red", label="Dataset 2")

plt.title("Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")

plt.legend()

plt.show()

# Matplotlib Q4

fig, axes = plt.subplots(1, 2)

# Left subplot (line plot)
axes[0].plot(x, y)
axes[0].set_title("Squares")

# Right subplot (bar plot)
axes[1].bar(subjects, scores)
axes[1].set_title("Subject Scores")

plt.tight_layout()

plt.show()

# --- Descriptive Statistics ---

# Descriptive Stats Q1

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

print("\nDescriptive statistics:")
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Standard deviation: {np.std(data)}")

# Descriptive Stats Q2

scores = np.random.normal(65, 10, 500)

plt.figure()

plt.hist(scores, bins=20)

plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")

plt.show()

# Descriptive Stats Q3

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.figure()

plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])

plt.title("Score Comparison")

plt.show()

# Descriptive Stats Q4

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.figure()

plt.boxplot(
    [normal_data, skewed_data],
    tick_labels=["Normal", "Exponential"]
)

plt.title("Distribution Comparison")

plt.show()

# The exponential distribution is more skewed than the normal distribution.
# The mean is a good choice for the normal distribution.
# The median is better for the exponential distribution because it is not affected as much by extreme values.

# Descriptive Stats Q5

import statistics

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print("\nData 1:")
print(f"Mean: {np.mean(data1)}")
print(f"Median: {np.median(data1)}")
print(f"Mode: {statistics.mode(data1)}")

print("\nData 2:")
print(f"Mean: {np.mean(data2)}")
print(f"Median: {np.median(data2)}")
print(f"Mode: {statistics.mode(data2)}")

# The value 150 is much larger than the other numbers.
# It pulls the mean up, but the median stays about the same.

# --- Hypothesis Testing ---

from scipy import stats

# Hypothesis Q1

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_value = stats.ttest_ind(group_a, group_b)

print("\nIndependent Samples t-test:")
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# Hypothesis Q2

alpha = 0.05

if p_value < alpha:
    print("\nThe result is statistically significant.")
else:
    print("\nThe result is not statistically significant.")

# Hypothesis Q3

before = [60, 65, 70, 58, 62, 67, 63, 66]
after = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_value = stats.ttest_rel(before, after)

print("\nPaired Samples t-test:")
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# Hypothesis Q4

scores = [72, 68, 75, 70, 69, 74, 71, 73]

t_stat, p_value = stats.ttest_1samp(scores, 70)

print("\nOne-Sample t-test:")
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# Hypothesis Q5

t_stat, p_value = stats.ttest_ind(
    group_a,
    group_b,
    alternative="less"
)

print("\nOne-tailed t-test:")
print(f"p-value: {p_value}")

# Hypothesis Q6
print("\nConclusion:")
print("Group A scored lower than Group B, and this difference is probably not due to chance.")


# --- Correlation ---

import seaborn as sns

# Correlation Q1

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

correlation_matrix = np.corrcoef(x, y)

print("\nCorrelation matrix:")
print(correlation_matrix)

print(f"Correlation coefficient: {correlation_matrix[0, 1]}")

# I expect the correlation to be 1 because y increases at the same rate as x.

# Correlation Q2

from scipy.stats import pearsonr

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]

correlation, p_value = pearsonr(x, y)

print("\nPearson correlation:")
print(f"Correlation coefficient: {correlation}")
print(f"p-value: {p_value}")


# Correlation Q3

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age": [25, 30, 22, 35, 28]
}

df = pd.DataFrame(people)

correlation_matrix = df.corr()

print("\nCorrelation matrix:")
print(correlation_matrix)

# Correlation Q4

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.figure()

plt.scatter(x, y)

plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Correlation Q5

plt.figure()

sns.heatmap(correlation_matrix, annot=True)

plt.title("Correlation Heatmap")

plt.show()

# --- Pipelines ---

# Pipeline Q1

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary

def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

result = data_pipeline(arr)

print("\nPipeline summary:")
for key, value in result.items():
    print(f"{key}: {value}")