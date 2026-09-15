# Video: https://youtu.be/y4wptQiDBnM
from openai import OpenAI
import joblib
import pandas as pd
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

# --- Step 1: Incremental Read ---
load_dotenv()

# Load the Week 4 model metadata
models_dir = Path(__file__).parent / "models"
metadata_path = models_dir / "weather_classifier_metadata.json"

with open(metadata_path, "r", encoding="utf-8") as file:
    model_metadata = json.load(file)

# Connect to Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Fetch all raw weather records
raw_response = supabase.table("weather_raw").select("*").execute()
raw_records = raw_response.data

# Fetch dates that are already enriched
enriched_response = (
    supabase.table("weather_enriched")
    .select("date")
    .execute()
)
enriched_dates = {record["date"] for record in enriched_response.data}

# Keep only records that have not been processed
records_to_process = [
    record
    for record in raw_records
    if record["date"] not in enriched_dates
]

print(f"Raw records: {len(raw_records)}")
print(f"Already enriched: {len(enriched_dates)}")
print(f"Records to process this run: {len(records_to_process)}")


# --- Step 2: ML Transform ---

# Load the saved weather classifier
model_path = models_dir / "weather_classifier.pkl"
weather_classifier = joblib.load(model_path)

feature_names = model_metadata["feature_names"]
enrichment_records = []

if records_to_process:
    # Select the model features in the same order used during training
    weather_df = pd.DataFrame(records_to_process)
    model_inputs = weather_df[feature_names]

    predictions = weather_classifier.predict(model_inputs)
    probabilities = weather_classifier.predict_proba(model_inputs)[:, 1]

    # Build the enrichment records
    enrichment_records = [
        {
            "date": record["date"],
            "good_for_running": bool(prediction),
            "confidence": float(probability),
        }
        for record, prediction, probability in zip(
            records_to_process,
            predictions,
            probabilities,
        )
    ]

    good_days = int(sum(predictions))
    minimum_confidence = float(probabilities.min())
    maximum_confidence = float(probabilities.max())

    print(f"Days classified as good: {good_days}")
    print(
        f"Confidence range: "
        f"{minimum_confidence:.4f} to {maximum_confidence:.4f}"
    )
else:
    print("Days classified as good: 0")
    print("Confidence range: No records to process")


# --- Step 3: LLM Transform ---

system_prompt = """
You write short running recommendations based on weather data and an ML
prediction. Write exactly one sentence that agrees with the ML prediction
and briefly explains the reason. Do not include a heading or extra text.
"""


def build_user_message(weather_record, enrichment_record):
    prediction = (
        "good for running"
        if enrichment_record["good_for_running"]
        else "skip running"
    )

    return f"""
Date: {weather_record["date"]}
Maximum temperature: {weather_record["temperature_2m_max"]} C
Minimum temperature: {weather_record["temperature_2m_min"]} C
Precipitation: {weather_record["precipitation_sum"]} mm
Maximum wind speed: {weather_record["wind_speed_10m_max"]} km/h
ML prediction: {prediction}
Probability of good running weather: {enrichment_record["confidence"]:.2%}
"""


def create_llm_summary(client, weather_record, enrichment_record):
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


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

for index, (weather_record, enrichment_record) in enumerate(
    zip(records_to_process, enrichment_records),
    start=1,
):
    enrichment_record["llm_summary"] = create_llm_summary(
        openai_client,
        weather_record,
        enrichment_record,
    )

    if index % 50 == 0:
        print(f"Processed {index} of {len(enrichment_records)} records")

# --- Step 4: Load ---

if enrichment_records:
    supabase.table("weather_enriched").upsert(
        enrichment_records,
        on_conflict="date",
    ).execute()

print(f"Rows upserted: {len(enrichment_records)}")


# --- Step 5: Verify ---

verification_response = (
    supabase.table("weather_enriched")
    .select("date, good_for_running, confidence, llm_summary")
    .order("date")
    .execute()
)
verified_records = verification_response.data

print(f"Total enriched rows: {len(verified_records)}")
print("Five sample rows:")

for record in verified_records[:5]:
    print(
        f'Date: {record["date"]}, '
        f'Good for running: {record["good_for_running"]}, '
        f'Confidence: {record["confidence"]:.4f}, '
        f'Summary: {record["llm_summary"]}'
    )

good_running_days = sum(
    record["good_for_running"] for record in verified_records
)
print(f"Total days classified as good: {good_running_days}")

# The summaries I checked generally match the weather conditions and the
# model's prediction. I think the January 5 summary is especially good
# because it supports the good-for-running prediction by mentioning the mild
# temperatures and lack of rain. The January 2 summary seems weaker because
# the confidence is only 0.5063, but the recommendation sounds very certain.
# This may have happened because my prompt told the LLM to follow the
# prediction but did not tell it to use cautious wording when the confidence
# is close to 0.5.


# --- Step 6: Reflect ---
#
# My classifier was actually trained with Piscataway, NJ weather data instead
# of Charlotte data. Since my Week 9 records also use Piscataway weather, I
# expect the predictions to be more relevant to the data being processed. If
# I used weather from a different city, the predictions might be less accurate
# because that city could have a different climate and weather patterns. In
# that case, I would retrain the classifier with data from the new city.
#
# The LLM does not override the classifier because my prompt tells it to
# follow the model's prediction. It only adds a written explanation. This
# means that if the classifier is wrong, the LLM could make the wrong
# prediction sound convincing instead of correcting it.
#
# If I processed 50,000 records, my main concern would be the cost of making
# 50,000 LLM calls, although the time required would also be a concern. I
# would continue skipping completed records, process the data in batches, and
# use controlled parallel requests to reduce the total processing time.