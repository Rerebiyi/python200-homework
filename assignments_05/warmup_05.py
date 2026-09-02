import json

# --- Completions API ---

# API Question 1

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?"
        }
    ]
)

print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Total tokens:", response.usage.total_tokens)


# API Question 2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temperature in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )

    print("Temperature:", temperature)
    print("Response:", response.choices[0].message.content)
    print()

# Higher temperatures gave more creative answers.
# I would use temperature 0 for more consistent results.


# API Question 3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
        }
    ],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices, start=1):
    print(f"Completion {i}:", choice.message.content)


# API Question 4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work."
        }
    ],
    max_tokens=15
)

print("Max Tokens Response:", response.choices[0].message.content)

# The response was cut short because max_tokens was set to 15.
# max_tokens can be used to keep responses short.


# --- System Messages and Personas ---

# System Question 1

messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Patient Tutor Response:", response.choices[0].message.content)

messages = [
    {
        "role": "system",
        "content": "You are a funny Python tutor. You explain things using jokes and humor."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Funny Tutor Response:", response.choices[0].message.content)

# The first response was patient and encouraging.
# The second response used more jokes and humor.

# System Question 2

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Conversation Response:", response.choices[0].message.content)

# The model knows Jordan's name because it was included in the conversation history.

# --- Prompt Engineering ---

# Prompt Question 1 - Zero-Shot

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Classify the sentiment of this review as positive, negative, or mixed: {review}"
            }
        ]
    )

    print(f"Review {i}:", response.choices[0].message.content)

# Prompt Question 2 - One-Shot

example = """
Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
"""

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""{example}

Classify the sentiment of this review as positive, negative, or mixed.

Review: "{review}"
Sentiment:"""
            }
        ]
    )

    print(f"Review {i}:", response.choices[0].message.content)
# The example made the responses shorter and more consistent.
# The model followed the format from the example.

# Prompt Question 3 - Few-Shot

examples = """
Example 1:
Review: "The service was excellent and the staff was friendly."
Sentiment: positive

Example 2:
Review: "The product stopped working after one day."
Sentiment: negative

Example 3:
Review: "The price was great, but shipping took too long."
Sentiment: mixed
"""

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""{examples}

Classify the sentiment of this review as positive, negative, or mixed.

Review: "{review}"
Sentiment:"""
            }
        ]
    )

    print(f"Review {i}:", response.choices[0].message.content)

# I would use zero-shot for simple tasks.
# I would use one-shot to show the model what I want.
# I would use few-shot when the model needs more examples.

# Prompt Question 4 - Chain of Thought

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": """Solve this problem and show your reasoning step by step before giving the final answer.

A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?

Label the final answer clearly."""
        }
    ]
)

print("Chain of Thought Response:", response.choices[0].message.content)

# Breaking the problem into steps makes it easier to get the right answer.


# Prompt Question 5 - Structured Output

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": f"""Analyze the review below.

Respond ONLY with valid JSON.
Do not include markdown, code fences, explanations, or any text outside the JSON object.

Use exactly these keys:
sentiment
confidence (a float from 0 to 1)
reason (one sentence)

Review: {review}"""
        }
    ]
)

raw_response = response.choices[0].message.content

print("Raw Response:", raw_response)

try:
    result = json.loads(raw_response)
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
except (json.JSONDecodeError, KeyError):
    print("The response was not valid JSON.")
    print("Raw Response:", raw_response)


# Prompt Question 6 - Delimiters

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("Instructions Response:", response.choices[0].message.content)

user_text = "Python is a popular programming language. It is used in many areas of technology."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("Non-Instructions Response:", response.choices[0].message.content)

# Delimiters help the model separate the instructions from the user's text.

# --- Local Models with Ollama ---
# Ollama Question 1

"""
Ollama Output:
A large language model is an AI system trained on massive datasets to understand and generate
human-like text, enabling tasks like answering questions or writing articles. It powers
applications by analyzing text to learn patterns and improve performance over time.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences."
        }
    ]
)

print("OpenAI Response:", response.choices[0].message.content)

# Both responses explained the same idea, but the OpenAI answer was a little more detailed.
# One advantage of a local model is that it can run without sending data to an external API.
# One disadvantage is that local models can be slower or less powerful.

