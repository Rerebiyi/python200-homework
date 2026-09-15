# Video link: https://youtu.be/3GCg_t3vW1g

import os
from datetime import date

import requests
from dotenv import load_dotenv
from supabase import create_client



# --- Step 1: Extract ---

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


def extract_weather_data():
    response = requests.get(url, params=params)
    response.raise_for_status()

    weather_data = response.json()
    daily_data = weather_data["daily"]

    print(f"Received {len(daily_data['time'])} days of weather data.")
    print(f"Date range: {daily_data['time'][0]} to {daily_data['time'][-1]}")
    print(f"Daily units: {weather_data['daily_units']}")

    return weather_data


weather_data = extract_weather_data()


# --- Step 2: Transform ---


def transform_weather_data(weather_data):
    daily_data = weather_data["daily"]
    records = []

    for index, weather_date in enumerate(daily_data["time"]):
        record = {
            "date": weather_date,
            "temperature_2m_max": daily_data["temperature_2m_max"][index],
            "temperature_2m_min": daily_data["temperature_2m_min"][index],
            "precipitation_sum": daily_data["precipitation_sum"][index],
            "wind_speed_10m_max": daily_data["wind_speed_10m_max"][index],
        }
        records.append(record)

    return records


records = transform_weather_data(weather_data)

print("First record:", records[0])
print("Last record:", records[-1])

# I expected 365 records because 2023 was not a leap year, and I got
# 365 records. If the numbers were different, the API could be missing
# some dates or the date range in my request could be wrong.


# --- Step 3: Load ---


def get_client():
    load_dotenv(override=True)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url:
        raise ValueError("SUPABASE_URL is missing from the .env file.")

    if not supabase_key:
        raise ValueError("SUPABASE_KEY is missing from the .env file.")

    return create_client(supabase_url, supabase_key)


def load_weather_data(supabase, records):
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print(f"Upserted {len(response.data)} rows.")


supabase = get_client()
load_weather_data(supabase, records)

# The row count stayed at 366 after I ran the script again. This shows
# that the pipeline is idempotent because it updated the same dates
# instead of adding duplicate rows.


# --- Step 4: Verify ---


def verify_weather_data(supabase):
    response = (
        supabase.table("weather_raw")
        .select("*", count="exact")
        .order("date")
        .execute()
    )

    records = response.data

    print(f"Total rows in weather_raw: {response.count}")
    print(f"Earliest date: {records[0]['date']}")
    print(f"Latest date: {records[-1]['date']}")

    target_date = date.fromisoformat("2023-07-04")

    july_fourth = next(
        (record for record in records if record["date"] == "2023-07-04"),
        None,
    )

    if july_fourth:
        print(f"Record for 2023-07-04: {july_fourth}")
    else:
        nearest_record = min(
            records,
            key=lambda record: abs(
                date.fromisoformat(record["date"]) - target_date
            ),
        )
        print(f"Nearest record to 2023-07-04: {nearest_record}")

verify_weather_data(supabase)
