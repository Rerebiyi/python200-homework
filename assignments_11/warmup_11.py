from prefect import get_run_logger, task

# Prefect Orchestration

# Prefect Question 1
#
# A task is one step in a pipeline. A flow connects the tasks and controls
# how the full pipeline runs.
#
# I would not use @task for the temperature function because it is only a
# simple calculation. It does not need retries or separate monitoring, so
# a regular Python function is enough.

# Prefect Question 2
#
# @task(name="call_api", retries=3, retry_delay_seconds=30)

# Prefect Question 3
#
# I would open the transform task-run page and check the Logs tab. I would
# look for the error message or traceback to see where the task failed.
# The timeline would also show that load_enriched did not run because the
# transform task failed.


# Production Patterns

# Production Question 1
#
# raise_for_status() raises an HTTPError when an API request fails. This
# tells Prefect that the task failed and stops the next tasks from running.
#
# If the API returns a 500 error and the code only prints "error," Prefect
# may think the task worked. The next tasks could then run with bad or
# missing data.


# Production Question 2
#
# Upsert prevents duplicate rows when the pipeline runs again. If a date
# already exists, it updates that row instead of adding another one.
#
# A regular insert could create duplicate rows or fail if the date must
# be unique.


# Production Question 3

@task
def load_enriched(enrichment_records: list):
    get_run_logger().info(f"Upserted {len(enrichment_records)} enrichment records")

# Production Question 4
#
# The incremental check skips dates that are already in weather_enriched.
# This prevents repeated LLM calls and database writes.
#
# Without it, all 365 records would be processed every time. The pipeline
# would cost more, take longer, and could replace saved summaries with
# different ones.
#
# Without it, all 365 records would go through the ML model and LLM every
# time. The pipeline would take longer, use more money, and could overwrite
# old recommendations with different ones.
