Cloud Concepts Question 1

Cloud computing means renting computing resources and paying for what is being used. This is different from owning servers because you do not have to buy or manage them yourself.

Cloud Concepts Question 2

Vertical scaling means giving one computer more power. For example, I could use a computer with more RAM or a faster GPU. Horizontal scaling means adding more computers to help with the work.

For the web app, I would use horizontal scaling so more computers can handle the extra users.

For the model training, I would use vertical scaling to get a computer with a faster GPU and more RAM.

For the data pipeline, I would use horizontal scaling because the files can be divided between multiple computers.


Cloud Concepts Question 3

Gmail: SaaS
Gmail is an app that is ready to use, and Google manages it for you.

Azure Virtual Machines: IaaS
Azure gives you a virtual computer, but you are responsible for setting it up and managing it.

AWS S3: IaaS
AWS S3 gives you online storage for your files and data.

GitHub Codespaces: PaaS
GitHub Codespaces gives developers an online place to write and run code.

Snowflake: SaaS
Snowflake is a ready-to-use service for storing and working with data.

Supabase: BaaS
Supabase gives developers ready-to-use backend tools, such as a database and user login.

IaaS

IaaS gives you basic resources like virtual computers and storage. Azure Virtual Machines is an example. The developer manages the operating system, software, and settings.

PaaS

PaaS gives you a place to build and run your code. GitHub Codespaces is an example. The developer manages the code, while the provider manages the platform.

SaaS

SaaS is a complete app that is ready to use. Gmail is an example. The provider manages everything, and the user just uses the app.


Cloud Concepts Question 4

A managed data platform like Databricks or Snowflake handles much of the setup needed to work with data. With AWS or GCP, you have to choose and set up the services yourself. A managed platform saves time, but it may cost more and gives you less control.

Cloud Concepts Question 5

The cloud may not be the best choice when the data and work can easily be handled on one computer. In that situation, using a local computer may be faster and cost less.

Part 2 Warmup - Cloud Landscape

Cloud Landscape Question 1

The three hyperscalers are AWS, GCP, and Microsoft Azure.

AWS offers a wide range of cloud services. It is often used by startups, large companies, and nonprofits with engineering teams.

GCP is strong in data and machine learning. It is often used by organizations that work with large amounts of data or machine learning models.

Microsoft Azure works well with other Microsoft products. It is often used by large companies and government organizations that already use Microsoft services.


Cloud Landscape Question 2

The first reason is that Supabase is easier to access. Students can make their own account in a few minutes, while Azure requires more steps before they can start.

The second reason is that Supabase stores data in rows and columns. This gives students more practice working with databases, which is useful for data jobs.

The third reason is that Supabase makes the pipeline easier to understand. The raw data and updated data are stored in separate tables, so students can easily check their work and find problems.

This suggests that I should choose a cloud tool based on what the project needs, how easy it is to use, and whether I can access it.


Cloud Landscape Question 3

For the image files, I would use object storage, such as AWS S3.

For the ML training job, I would use compute, such as Google Compute Engine with a GPU.

For the web API, I would use serverless compute, such as Azure Functions.

For sending data to a large language model, I would use an LLM API, such as Azure OpenAI.


Cloud Landscape Question 4

I would build a project that tracks raw materials that are close to expiring and matches them with upcoming production orders. I would store the data in Supabase, use AWS Lambda to check it each day, and use Azure OpenAI to explain which materials should be used first.

Using one provider could make the project easier to manage and keep the costs in one place. However, I would have fewer choices when deciding which service works best for each part