# Video: https://youtu.be/eO9igsD2XnY

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from prefect import flow, task
from supabase import create_client

def get_supabase_client():
    env_path = Path(__file__).with_name(".env")
    load_dotenv(env_path, override=True)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url:
        raise ValueError("SUPABASE_URL is missing from the .env file.")

    if not supabase_key:
        raise ValueError("SUPABASE_KEY is missing from the .env file.")

    return create_client(supabase_url, supabase_key)

def build_user_message(weather_record, enrichment_record):
    prediction = (
        "good for running"
        if enrichment_record["good_for_running"]
        else "skip running"
    )

    return (
        f'Date: {weather_record["date"]}\n'
        f'Maximum temperature: {weather_record["temperature_2m_max"]} C\n'
        f'Minimum temperature: {weather_record["temperature_2m_min"]} C\n'
        f'Precipitation: {weather_record["precipitation_sum"]} mm\n'
        f'Maximum wind speed: {weather_record["wind_speed_10m_max"]} km/h\n'
        f"ML prediction: {prediction}\n"
        f'Probability of good running weather: '
        f'{enrichment_record["confidence"]:.2%}'
    )


def create_llm_summary(client, weather_record, enrichment_record):
    system_prompt = (
        "Write exactly one sentence that gives a short running recommendation "
        "based on the weather data and agrees with the ML prediction. "
        "Do not include a heading or extra text."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": build_user_message(
                        weather_record,
                        enrichment_record,
                    ),
                },
            ],
        )

        return response.choices[0].message.content.strip()
    except Exception as error:
        print(f'LLM error for {weather_record["date"]}: {error}')
        return "A recommendation could not be generated for this day."



@task(retries=2, retry_delay_seconds=10)
def extract():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 40.59,
        "longitude": -74.46,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
        "timezone": "America/New_York",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    daily = response.json()["daily"]
    records = []

    for index, date in enumerate(daily["time"]):
        records.append(
            {
                "date": date,
                "temperature_2m_max": daily["temperature_2m_max"][index],
                "temperature_2m_min": daily["temperature_2m_min"][index],
                "precipitation_sum": daily["precipitation_sum"][index],
                "wind_speed_10m_max": daily["wind_speed_10m_max"][index],
            }
        )

    print(f"Extracted {len(records)} weather records.")
    return records

@task(retries=2, retry_delay_seconds=5)
def load_raw(raw_records):
    supabase = get_supabase_client()

    response = (
        supabase.table("weather_raw")
        .upsert(raw_records, on_conflict="date")
        .execute()
    )

    print(f"Upserted {len(response.data)} rows into weather_raw.")
    return raw_records

@task
def transform(raw_records):
    supabase = get_supabase_client()

    enriched_response = (
        supabase.table("weather_enriched")
        .select("date")
        .execute()
    )
    enriched_dates = {
        record["date"] for record in enriched_response.data
    }

    records_to_process = [
        record
        for record in raw_records
        if record["date"] not in enriched_dates
    ]

    print(f"Already enriched: {len(enriched_dates)}")
    print(f"Records to process: {len(records_to_process)}")

    models_dir = Path(__file__).parent / "models"
    model_path = models_dir / "weather_classifier.pkl"
    metadata_path = models_dir / "weather_classifier_metadata.json"

    weather_classifier = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        model_metadata = json.load(file)

    feature_names = model_metadata["feature_names"]
    enrichment_records = []

    if not records_to_process:
        print("No new records to transform.")
        return enrichment_records

    weather_df = pd.DataFrame(records_to_process)

    missing_features = [
        feature
        for feature in feature_names
        if feature not in weather_df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    model_inputs = weather_df[feature_names]
    predictions = weather_classifier.predict(model_inputs)
    probabilities = weather_classifier.predict_proba(model_inputs)[:, 1]

    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY is missing from the .env file.")

    openai_client = OpenAI(api_key=openai_key)

    for index, (weather_record, prediction, probability) in enumerate(
        zip(records_to_process, predictions, probabilities),
        start=1,
    ):
        enrichment_record = {
            "date": weather_record["date"],
            "good_for_running": bool(prediction),
            "confidence": float(probability),
        }

        enrichment_record["llm_summary"] = create_llm_summary(
            openai_client,
            weather_record,
            enrichment_record,
        )
        enrichment_records.append(enrichment_record)

        if index % 50 == 0:
            print(
                f"Processed {index} of "
                f"{len(records_to_process)} records."
            )

    return enrichment_records


@task(retries=2, retry_delay_seconds=5)
def load_enriched(enrichment_records):
    if not enrichment_records:
        print("No enrichment records to load.")
        return

    supabase = get_supabase_client()

    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )

    print(
        f"Upserted {len(response.data)} rows into weather_enriched."
    )

@flow(log_prints=True)
def weather_etl_pipeline():
    raw_records = extract()
    loaded_records = load_raw(raw_records)
    enrichment_records = transform(loaded_records)
    load_enriched(enrichment_records)

    print("Weather ETL pipeline completed.")


if __name__ == "__main__":
    weather_etl_pipeline()
