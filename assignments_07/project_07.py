import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import os

from dotenv import load_dotenv
import pandas as pd
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool


# --- Task 1 ---

df = None

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

df = None

DATA_PATH = REPO_ROOT / "assignments_01/outputs/merged_happiness.csv"
FALLBACK_DIR = REPO_ROOT / "assignments/resources/happiness_project"


@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Loads the merged World Happiness CSV if it exists. If the merged file
    is not found, loads and combines the yearly CSV files from the fallback
    directory. The loaded data is stored in the shared global DataFrame.

    Returns:
        A dict containing the shape of the dataset and its column names.
    """
    global df

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)

    else:
        csv_files = sorted(FALLBACK_DIR.glob("*.csv"))

        if not csv_files:
            return {"error": "No World Happiness CSV files were found."}

        yearly_data = []

        for file in csv_files:
            yearly_df = pd.read_csv(file)

            if "year" not in yearly_df.columns:
                year = int(file.stem[:4])
                yearly_df["year"] = year

            yearly_data.append(yearly_df)

        df = pd.concat(yearly_data, ignore_index=True)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
    }


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for one column in the loaded dataset.

    Args:
        column: The name of the column to summarize.

    Returns:
        A dict containing descriptive statistics for the column, or an
        error dict if no data is loaded or the column does not exist.
    """
    if df is None:
        return {"error": "No data is loaded."}

    if column not in df.columns:
        return {"error": f"Column '{column}' was not found."}

    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two numeric columns.

    Args:
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.

    Returns:
        A dict containing both column names, the Pearson correlation
        coefficient, and the p-value, or an error dict for bad input.
    """
    if df is None:
        return {"error": "No data is loaded."}

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns were not found."}

    if (
        not pd.api.types.is_numeric_dtype(df[col1])
        or not pd.api.types.is_numeric_dtype(df[col2])
    ):
        return {"error": "Both columns must be numeric."}

    clean_data = df[[col1, col2]].dropna()

    if len(clean_data) < 2:
        return {"error": "Not enough valid data to compute correlation."}

    pearson_r, p_value = pearsonr(
        clean_data[col1],
        clean_data[col2],
    )

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(pearson_r), 4),
        "p_value": round(float(p_value), 4),
    }


@tool
def get_top_n_countries(
    column: str,
    year: int,
    n: int = 5,
) -> list[dict] | dict:
    """Return the top N countries for a column in a specific year.

    Args:
        column: The column used to rank the countries.
        year: The year to filter the dataset by.
        n: The number of top countries to return.

    Returns:
        A list containing the top countries as records with the country name
        and requested column value, or an error dict for bad input.
    """
    if df is None:
        return {"error": "No data is loaded."}

    if column not in df.columns:
        return {"error": f"Column '{column}' was not found."}

    if "year" not in df.columns or "Country" not in df.columns:
        return {"error": "Required year or Country column was not found."}

    year_data = df[df["year"] == year]

    if year_data.empty:
        return {"error": f"No data was found for year {year}."}

    if n <= 0:
        return {"error": "n must be greater than 0."}

    top_countries = (
        year_data
        .sort_values(by=column, ascending=False)
        .head(n)[["Country", column]]
        .rename(columns={"Country": "country"})
        .to_dict(orient="records")
    )

    return top_countries


# --- Task 2 ---

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini",
)


SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"When writing Python code for analysis or plotting, use the preloaded DataFrame named happiness_df. Do not create simulated or placeholder data."
"""


agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries,
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "scipy.stats",
        "os",
    ],
    max_steps=8,
)


if __name__ == "__main__":

    # Run relative output paths from the Assignment 7 folder.
    os.chdir(BASE_DIR)

    # --- Task 3 ---

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    # Load the DataFrame and make it available to the CodeAgent's Python executor.
    load_happiness_data()
    agent.python_executor.send_variables({"happiness_df": df})
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)


    # --- Task 4 ---

    # My query 1
    my_query_1 = (
        "Which 5 countries had the highest healthy life expectancy in 2019?"
    )

    response_1 = agent.run(
        my_query_1,
        reset=False,
    )

    print(response_1)

    # Comment: This triggered tool use with get_top_n_countries, not code generation.


    # My query 2
    my_query_2 = (
        "Create a scatter plot comparing social support and happiness score for 2024. "
        "Save it to outputs/social_support_vs_happiness.png."
    )

    response_2 = agent.run(
        my_query_2,
        reset=False,
    )

    print(response_2)

    # Comment: This triggered code generation, not tool use.


    # --- Task 5: Reflection ---
    #
    # 1. The agent said the correlation was statistically significant because the
    #    p-value was 0.0. It used the p-value correctly, but it did not clearly state
    #    the threshold. A common threshold for significance is 0.05.
    #
    # 2. I was surprised that the agent could create plots by writing its own code.
    #    For example, it created and saved a scatter plot without using a plotting tool.
    #
    # 3. I would add a tool that compares a country across different years.
    #    It could show how happiness scores changed over time and help find which
    #    countries improved or declined the most.