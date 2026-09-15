import time

# --- ML vs. LLM in Pipelines ---

# ML/LLM Question 1
#
# In this pipeline, my Week 4 ML classifier looks at the weather data and
# predicts whether the day is good for running or should be skipped. It also
# gives a confidence score for its prediction. The classifier handles this
# part because it was trained on weather data to make consistent predictions.
#
# The LLM takes the weather details, the prediction, and the confidence score
# and turns them into a short recommendation that is easy to understand. The
# LLM handles this part because it is better at writing natural-language
# explanations.
#
# If I swapped their roles, the LLM might not make the good/skip decision
# as consistently as my trained classifier. The ML model also could not write
# the recommendation because it only returns a prediction and probability,
# not a complete sentence.


# ML/LLM Question 2
#
# I would use deterministic code to convert "2023-07-04" to a day of the
# week because date libraries can calculate the exact answer.
#
# I would use an LLM to classify a job posting because it can understand
# the meaning and context of freeform text.
#
# I would use a trained ML model to predict customer churn because there is
# labeled training data and numeric features it can learn patterns from.
#
# I would use deterministic code to normalize city names because I can map
# each known variation to one standard city name.
#
# I would use deterministic code to sum revenue because it is a direct
# calculation that does not require a prediction or written response.


# ML/LLM Question 3
#
# Incremental processing means I only process dates that are not already in
# weather_enriched. This prevents my script from doing the same work every
# time it runs. If I processed all 365 records again, I would pay for 365
# unnecessary LLM calls. I could also create duplicate records or replace
# recommendations that were already saved. Skipping completed dates saves
# money and keeps my data accurate.


# --- Prompt Design ---

# Prompt Question 1

system_prompt = """
Write exactly two sentences about whether the weather is good for running.
In the first sentence, state the model's prediction.
In the second sentence, explain the prediction using the weather conditions
and confidence score. Do not include any extra text.
"""

# I would change my validation logic to require exactly two sentences instead
# of one. I would also check that both sentences contain text before accepting
# the response.


# Prompt Question 2

def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


# I would use this in a production pipeline when a temporary API or network
# error happens. Retrying gives the request another chance to work without
# stopping the entire pipeline.
