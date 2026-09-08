## Supabase Setup

I set up my Supabase project and confirmed that the weather_raw and weather_enriched tables have the correct columns.

## Cloud Cost Analysis

Scenario A costs $1.66 per month. Scenario B costs $2,384.48 per month. The GPU costs $2,233.80, the database costs $127.13, and the storage costs $23.55. I was surprised that the GPU cost so much more than everything else. I also tried a larger SQL Server database with extra settings, and the cost went over $6,000 per month. A GPU can be useful when training a large model, but running one all month would cost too much for a smaller project.


## Video

[Week 8 Project Video](https://youtu.be/oNzFL2Zhx2g)
