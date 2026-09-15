import os
from datetime import date

from dotenv import load_dotenv
from supabase import create_client



# --- Supabase Connection ---

# Question 1
# supabase-py needs my project URL and API key to connect to Supabase.
# I can find both in the API section of my Supabase project settings.
# I should not put them directly in my Python code because someone
# could see them if I share the code or push it to GitHub. Instead,
# I should keep them in a .env file.


# Question 2
def get_client():
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url:
        raise ValueError("SUPABASE_URL is missing from the .env file.")

    if not supabase_key:
        raise ValueError("SUPABASE_KEY is missing from the .env file.")

    return create_client(supabase_url, supabase_key)

# Question 3
# Row Level Security controls which database rows a user can view or
# change. We disabled it for this course so our Python scripts can read
# and write data without setting up security policies. In a real app
# with different users and private information, I would keep it enabled
# so users can only access the data they are allowed to use.


# --- supabase-py CRUD ---

# CRUD Question 1
def insert_test_record(supabase):
    test_record = {
        "date": date.today().isoformat(),
        "temperature_2m_max": 26.0,
        "temperature_2m_min": 17.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 14.0,
    }

    response = supabase.table("weather_raw").insert(test_record).execute()
    print(response.data)


supabase = get_client()
# insert_test_record(supabase)

# If I ran this function twice on the same day, the second insert would
# cause an error because that date already exists in the table. To make
# it safe to run multiple times, I would use upsert with
# on_conflict="date" instead of insert. This would update the row if
# the date already exists.



# CRUD Question 2
def get_records_by_date_range(supabase, start, end):
    response = (
        supabase.table("weather_raw")
        .select("*")
        .gte("date", start)
        .lte("date", end)
        .execute()
    )

    return response.data


records = get_records_by_date_range(
    supabase,
    "2026-09-10",
    "2026-09-10",
)
print(records)


# CRUD Question 3
# insert adds a new row to the table. I would use it when adding a new
# user's information that is not already saved.
#
# upsert adds a new row or updates a row that already exists. I would
# use it when running the weather pipeline again because some dates
# may already be in the table.

def safe_upsert(supabase, records):
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print(f"Rows affected: {len(response.data)}")


# --- Idempotency ---

# Idempotency Question 1
# A pipeline should be safe to run again if it stops.
# For example, the weather script could load six months of data and
# then crash. If I restart it using insert, it would get an error when
# it reaches dates that are already saved. Using upsert lets it update
# those dates and finish loading the rest of the data.
