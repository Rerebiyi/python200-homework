import json

# --- Task 1: Setup and System Prompt ---

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


system_prompt = """
You are a job application coach helping users improve their job application materials.

Stay focused on job application materials such as resumes, cover letters, and related application writing.

Always remind the user to review and edit your output before submitting it anywhere.

You may not know the specific norms or expectations of the user's industry, so remind the user to use their own judgment when deciding what to include.
"""

# I made the assistant stay focused on job applications so its responses stay relevant.



# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs.
    Do not add any results, outcomes, numbers, or details that are not explicitly stated in the original bullet.
    Keep all facts the same as the original.
    If the original does not include a result, do not invent one.
    Only improve the wording.
    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).
    Respond ONLY with valid JSON, no markdown, no backticks, and no other text.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)
  
    try:
        results = json.loads(response)

        for item in results:
            print("Original:", item["original"])
            print("Improved:", item["improved"])
            print()

        return results

    except json.JSONDecodeError:
        print("The response was not valid JSON.")
        print("Raw Response:", response)
        return []


bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)


# The original bullets are vague and use weak wording.
# The model made them more specific and used stronger action verbs.


# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
    Do not invent skills, results, experiences, motivations, achievements, or personal qualities.
    Only use facts that are directly stated in the background.
    Do not add claims about passion, impact, success, or abilities unless they are explicitly provided.
        Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages)


job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

cover_letter = generate_cover_letter(job_title, background)
print("Cover Letter Opening:", cover_letter)

# I chose these examples because they show career changers applying to new fields.
# Few-shot prompting helps keep the style and format consistent.
# The model can still add details that were not given.

# --- Task 4: Moderation Check ---

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("I can't process that message. Please rephrase it and try again.")
        return False

    return True

safe_text = "Can you help me improve my resume?"
flagged_text = "I want to hurt someone."

print("Safe Test:", is_safe(safe_text))
print("Flagged Test:", is_safe(flagged_text))


# --- Task 5: The Chatbot Loop ---

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue

        # 5. Check if the user wants to rewrite bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []

            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            bullet_text = "\n".join(raw_bullets)

            if not is_safe(bullet_text):
                continue

            results = rewrite_bullets(raw_bullets)

            messages.append({
                "role": "user",
                 "content": "Rewrite these resume bullets:\n" + bullet_text
            })

            messages.append({
                "role": "assistant",
                "content": str(results)
            })

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            if not is_safe(job_title) or not is_safe(background):
                continue

            cover_letter = generate_cover_letter(job_title, background)
            print("\nJob Application Helper:", cover_letter)

            messages.append({
                "role": "user",
                 "content": f"Write a cover letter opening for {job_title}. Background: {background}"
            })

            messages.append({
                "role": "assistant",
                "content": cover_letter
            })

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})

            reply = get_completion(messages)

            print("\nJob Application Helper:", reply)

            messages.append({"role": "assistant", "content": reply})

   


if __name__ == "__main__":
    run_chatbot()


# --- Task 6: Ethics Reflection ---
# Format: Option A - Comment block
#
# The bot could be biased toward certain writing styles, industries, or backgrounds.
# This could cause it to give better advice to some people than others.
# Users should review the output because the bot can make mistakes or add details that are not true.
# If I used this tool professionally, I would add a warning telling users to review everything before submitting it.