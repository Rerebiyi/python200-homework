import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from prefect import flow, task
from prefect.logging import get_run_logger

from scipy.stats import pearsonr, ttest_ind


# Task 1: Load all yearly happiness files and combine them into one dataset.
@task(retries=3, retry_delay_seconds=2)
def load_happiness_data():
    logger = get_run_logger()

    # Find the folder where this script is located.
    script_directory = Path(__file__).resolve().parent

    # Go to the folder that contains the happiness data.
    data_directory = (
        script_directory.parent
        / "assignments"
        / "resources"
        / "happiness_project"
    )

    # Create the list of years to load.
    years = range(2015, 2025)

    # Build the file path for each year's CSV file.
    file_paths = [
        data_directory / f"world_happiness_{year}.csv"
        for year in years
    ]

    # Store each DataFrame before combining them.
    dataframes = []

    # Read each file one at a time.
    for file_path in file_paths:
        df = pd.read_csv(
            file_path,
            sep=";",
            decimal=","
        )

        # Keep the happiness score column name the same for every year.
        df = df.rename(columns={"Ladder score": "Happiness score"})

        # Add the year so we know which file each row came from.
        year = int(file_path.stem.split("_")[-1])
        df["year"] = year

        dataframes.append(df)

    # Combine all years into one DataFrame.
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the merged dataset in the outputs folder.
    output_directory = script_directory / "outputs"
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / "merged_happiness.csv"

    merged_df.to_csv(output_path, index=False)

    logger.info(f"Merged dataset saved to {output_path}")

    return merged_df


# Task 2: Calculate descriptive statistics for the happiness scores.
@task
def compute_descriptive_stats(df):
    logger = get_run_logger()

    # Calculate the overall happiness statistics.
    mean_score = df["Happiness score"].mean()
    median_score = df["Happiness score"].median()
    std_score = df["Happiness score"].std()

    logger.info(f"Mean happiness score: {mean_score}")
    logger.info(f"Median happiness score: {median_score}")
    logger.info(f"Standard deviation: {std_score}")

    # Calculate the average happiness score for each year.
    yearly_mean = df.groupby("year")["Happiness score"].mean()

    logger.info("Mean happiness score by year:")
    logger.info(f"\n{yearly_mean}")

    # Calculate the average happiness score for each region.
    regional_mean = df.groupby("Regional indicator")["Happiness score"].mean()

    logger.info("Mean happiness score by region:")
    logger.info(f"\n{regional_mean}")


# Task 3: Create and save visualizations.
@task
def create_visualizations(df):
    logger = get_run_logger()

    script_directory = Path(__file__).resolve().parent
    output_directory = script_directory / "outputs"

    # Create a histogram of all happiness scores.
    plt.figure()

    plt.hist(df["Happiness score"], bins=20)

    plt.title("Happiness Score Distribution")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")

    histogram_path = output_directory / "happiness_histogram.png"

    plt.savefig(histogram_path)
    plt.close()

    logger.info(f"Saved {histogram_path}")

    # Create a boxplot of happiness scores by year.
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="year",
        y="Happiness score"
    )

    plt.title("Happiness Scores by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")

    boxplot_path = output_directory / "happiness_by_year.png"

    plt.savefig(boxplot_path)
    plt.close()

    logger.info(f"Saved {boxplot_path}")

    # Create a scatter plot of GDP per capita and happiness score.
    plt.figure()

    plt.scatter(
        df["GDP per capita"],
        df["Happiness score"]
    )

    plt.title("GDP per Capita vs. Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")

    scatter_path = output_directory / "gdp_vs_happiness.png"

    plt.savefig(scatter_path)
    plt.close()

    logger.info(f"Saved {scatter_path}")

    # Create a correlation heatmap for all numeric columns.
    numeric_columns = df.select_dtypes(include="number")

    correlation_matrix = numeric_columns.corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True
    )

    plt.title("Correlation Heatmap")

    heatmap_path = output_directory / "correlation_heatmap.png"

    plt.savefig(heatmap_path)
    plt.close()

    logger.info(f"Saved {heatmap_path}")


# Task 4: Perform hypothesis testing.
@task
def perform_hypothesis_tests(df):
    logger = get_run_logger()

    # Get the happiness scores for 2019 and 2020.
    happiness_2019 = df[df["year"] == 2019]["Happiness score"]
    happiness_2020 = df[df["year"] == 2020]["Happiness score"]

    # Calculate the mean happiness score for each year.
    mean_2019 = happiness_2019.mean()
    mean_2020 = happiness_2020.mean()

    # Perform an independent samples t-test.
    t_statistic, p_value = ttest_ind(
        happiness_2019,
        happiness_2020
    )

    logger.info(f"2019 mean happiness score: {mean_2019}")
    logger.info(f"2020 mean happiness score: {mean_2020}")

    logger.info(f"t-statistic: {t_statistic}")
    logger.info(f"p-value: {p_value}")

    # Interpret the results using alpha = 0.05.
    if p_value < 0.05:
        logger.info(
            "There is a statistically significant difference in happiness scores between 2019 and 2020. "
            "This suggests that average global happiness changed between the two years."
        )
    else:
        logger.info(
            "There is no statistically significant difference in happiness scores between 2019 and 2020. "
            "Based on this data, we do not have enough evidence to conclude that average global happiness changed."
        )

    # Compare North America and ANZ with Sub-Saharan Africa.
    north_america = df[
        df["Regional indicator"] == "North America and ANZ"
    ]["Happiness score"]

    sub_saharan_africa = df[
        df["Regional indicator"] == "Sub-Saharan Africa"
    ]["Happiness score"]

    # Calculate the mean happiness score for each region.
    north_america_mean = north_america.mean()
    sub_saharan_mean = sub_saharan_africa.mean()

    # Perform an independent samples t-test for the two regions.
    region_t_statistic, region_p_value = ttest_ind(
        north_america,
        sub_saharan_africa
    )

    logger.info(f"North America and ANZ mean happiness score: {north_america_mean}")
    logger.info(f"Sub-Saharan Africa mean happiness score: {sub_saharan_mean}")

    logger.info(f"t-statistic: {region_t_statistic}")
    logger.info(f"p-value: {region_p_value}")

    # Interpret the results using alpha = 0.05.
    if region_p_value < 0.05:
        logger.info(
            "There is a statistically significant difference in happiness scores between North America and ANZ and Sub-Saharan Africa. "
            "This suggests that the average happiness scores for these two regions are different."
        )
    else:
        logger.info(
            "There is no statistically significant difference in happiness scores between North America and ANZ and Sub-Saharan Africa. "
            "Based on this data, we do not have enough evidence to conclude that the average happiness scores are different."
        )

# Task 5: Calculate correlations and apply the Bonferroni correction.
@task
def analyze_correlations(df):
    logger = get_run_logger()

    # Select the numeric explanatory variables.
    explanatory_variables = [
        "GDP per capita",
        "Social support",
        "Healthy life expectancy",
        "Freedom to make life choices",
        "Generosity",
        "Perceptions of corruption"
    ]

    # Calculate the Bonferroni-adjusted significance level.
    number_of_tests = len(explanatory_variables)
    adjusted_alpha = 0.05 / number_of_tests

    logger.info(f"Number of correlation tests: {number_of_tests}")
    logger.info(f"Bonferroni-adjusted alpha: {adjusted_alpha}")

    # Calculate the Pearson correlation for each explanatory variable.
    for column in explanatory_variables:
        # Drop rows with missing values in either column, since pearsonr
        # cannot handle NaN values.
        valid_rows = df[[column, "Happiness score"]].dropna()

        correlation, p_value = pearsonr(
            valid_rows[column],
            valid_rows["Happiness score"]
        )

        logger.info(f"{column}:")
        logger.info(f"Correlation coefficient: {correlation}")
        logger.info(f"p-value: {p_value}")

        if p_value < 0.05:
            logger.info("Significant at alpha = 0.05")
        else:
            logger.info("Not significant at alpha = 0.05")

        if p_value < adjusted_alpha:
            logger.info("Significant after Bonferroni correction")
        else:
            logger.info("Not significant after Bonferroni correction")


# Task 6: Log a summary of the key findings.
@task
def create_summary_report(df):
    logger = get_run_logger()

    # Count the countries and years in the merged dataset.
    total_countries = df["Country"].nunique()
    total_years = df["year"].nunique()

    logger.info(
        f"The merged dataset contains {total_countries} "
        f"countries across {total_years} years."
    )

    # Calculate the average happiness score for each region.
    regional_means = (
        df.groupby("Regional indicator")[
            "Happiness score"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    top_regions = regional_means.head(3)
    bottom_regions = regional_means.tail(3)

    logger.info(
        "Top 3 regions by mean happiness score:\n"
        f"{top_regions}"
    )

    logger.info(
        "Bottom 3 regions by mean happiness score:\n"
        f"{bottom_regions}"
    )

    # Repeat the 2019 and 2020 comparison for the summary.
    happiness_2019 = df[
        df["year"] == 2019
    ]["Happiness score"]

    happiness_2020 = df[
        df["year"] == 2020
    ]["Happiness score"]

    mean_2019 = happiness_2019.mean()
    mean_2020 = happiness_2020.mean()

    t_statistic, p_value = ttest_ind(
        happiness_2019,
        happiness_2020
    )

    if p_value < 0.05:
        logger.info(
            "The average happiness score changed significantly "
            f"from {mean_2019:.3f} in 2019 to "
            f"{mean_2020:.3f} in 2020. "
            "The difference is unlikely to be due to chance."
        )
    else:
        logger.info(
            "The average happiness score increased slightly "
            f"from {mean_2019:.3f} in 2019 to "
            f"{mean_2020:.3f} in 2020, but the difference "
            "was not statistically significant."
        )

    # Find the strongest significant correlation after correction.
    explanatory_variables = [
        "GDP per capita",
        "Social support",
        "Healthy life expectancy",
        "Freedom to make life choices",
        "Generosity",
        "Perceptions of corruption"
    ]

    number_of_tests = len(
        explanatory_variables
    )
    adjusted_alpha = (
        0.05 / number_of_tests
    )

    significant_correlations = []

    for column in explanatory_variables:
        valid_rows = df[
            [column, "Happiness score"]
        ].dropna()

        correlation, correlation_p_value = pearsonr(
            valid_rows[column],
            valid_rows["Happiness score"]
        )

        if correlation_p_value < adjusted_alpha:
            significant_correlations.append(
                {
                    "variable": column,
                    "correlation": correlation,
                    "p_value": correlation_p_value
                }
            )

    if significant_correlations:
        strongest_result = max(
            significant_correlations,
            key=lambda result: abs(
                result["correlation"]
            )
        )

        logger.info(
            "The variable most strongly correlated with "
            "happiness after the Bonferroni correction was "
            f"{strongest_result['variable']} "
            f"(correlation = "
            f"{strongest_result['correlation']:.3f}, "
            f"p-value = "
            f"{strongest_result['p_value']})."
        )
    else:
        logger.info(
            "None of the explanatory variables remained "
            "statistically significant after the "
            "Bonferroni correction."
        )


@flow
def happiness_pipeline():
    merged_df = load_happiness_data()

    compute_descriptive_stats(
        merged_df
    )

    create_visualizations(
        merged_df
    )

    perform_hypothesis_tests(
        merged_df
    )

    analyze_correlations(
        merged_df
    )

    create_summary_report(
        merged_df
    )


if __name__ == "__main__":
    happiness_pipeline()