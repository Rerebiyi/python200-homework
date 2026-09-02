import numpy as np
import pandas as pd

from prefect import flow, task
from prefect.logging import get_run_logger


@task
def create_series(arr):
    logger = get_run_logger()

    series = pd.Series(arr, name="values")

    logger.info("Created the pandas Series.")
    return series


@task
def clean_data(series):
    logger = get_run_logger()

    cleaned_series = series.dropna()

    logger.info("Removed the missing values.")
    return cleaned_series


@task
def summarize_data(series):
    logger = get_run_logger()

    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

    logger.info(f"Mean: {summary['mean']}")
    logger.info(f"Median: {summary['median']}")
    logger.info(f"Standard deviation: {summary['std']}")
    logger.info(f"Mode: {summary['mode']}")
    return summary


@flow
def pipeline_flow():
    arr = np.array([
        12.0,
        15.0,
        np.nan,
        14.0,
        10.0,
        np.nan,
        18.0,
        14.0,
        16.0,
        22.0,
        np.nan,
        13.0
    ])

    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)

    return summary


if __name__ == "__main__":
    summary = pipeline_flow()
    print(summary)

# Prefect may be more work than needed here because the pipeline only has three small steps and uses a small amount of data.

# Prefect could be useful if the pipeline ran on a schedule, used larger datasets, needed retries after errors, or needed monitoring.