from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.readers.file import PyMuPDFReader


# --- RAG Concepts ---

# Concepts Question 1

# Scenario A: RAG
# RAG is best because it can use the company's PDFs to answer questions.
# It also works well since the PDFs are updated often.

# Scenario B: Fine-tuning
# Fine-tuning is best because they have many examples of the writing style
# they want the model to learn.

# Scenario C: Prompt engineering
# Prompt engineering is best because there is only one short report.
# The report can be added directly to the prompt.

# Concepts Question 2

# A wrong answer can be harmful if the person believes it is true.
# For example, wrong legal information could cause someone to make a bad choice.
# If the AI sounds confident, people may trust the answer more. 

# Concepts Q3

# steps = [
#     "Extract text from source documents",
#     "Split text into chunks",
#     "Convert text chunks into embeddings",
#     "Receive the user's query",
#     "Embed the user's query",
#     "Retrieve the most relevant chunks",
#     "Inject retrieved chunks into the prompt",
#     "Generate a response from the LLM",
# ]

# 1. Extract text from source documents - The system gets the text from the documents.
# 2. Split text into chunks - The system breaks the text into smaller pieces.
# 3. Convert text chunks into embeddings - The system turns each chunk into numbers that represent its meaning.
# 4. Receive the user's query - The system gets the question the user wants answered.
# 5. Embed the user's query - The system turns the question into numbers so it can compare meanings.
# 6. Retrieve the most relevant chunks - The system finds the chunks that best match the question.
# 7. Inject retrieved chunks into the prompt - The system adds the matching chunks to the prompt as context.
# 8. Generate a response from the LLM - The model uses the question and context to create the answer.

# --- Keyword RAG ---

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")
    scores.sort(key=lambda x: x[0], reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# Keyword Question 1

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# hours.txt was selected because it matched the word weekends.



# Keyword Question 2

query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# No document matched the query, so the function returned the fallback result "None found."
# Keyword RAG did not work well because the menu has drink options, but the words did not match.
# Semantic retrieval would work better because it can understand similar meanings instead of exact words.

# Keyword Question 3

# I predict no document will be selected because none of the documents use the word rewards.

query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# My prediction was correct because there were no matching keywords.


# --- Semantic RAG Concepts ---

# Semantic Question 1

# An embedding is a way to represent the meaning of text using numbers.

# I would choose the 0.85 chunk because it is a closer match to the query.
# A higher score means the texts are more related.

# Semantic search can find a match because it looks for similar meanings,
# even if the same words are not used.


# Semantic Question 2

# | Feature                 | Keyword RAG                         | Semantic RAG                       |
# |-------------------------|-------------------------------------|------------------------------------|
# | What is compared?       | Words in the question and document  | The meaning of the text            |
# | What is retrieved?      | Documents with matching words       | Chunks with similar meaning        |
# | Can it handle synonyms? | Not very well                       | Yes                                |
# | Storage format          | Text stored in a dictionary         | Embeddings stored in a vector index |
# | Relevance score         | How many words match                | How similar the meanings are       |

# --- LlamaIndex ---

# LlamaIndex Question 1

pdf_path = "../../python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs"

docs = SimpleDirectoryReader(
    pdf_path,
    file_extractor={".pdf": PyMuPDFReader()}
).load_data()

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    print(f"\nQuestion: {question}")

    response = query_engine.query(question)
    print("Answer:", response)

    for node_with_score in response.source_nodes:
        print(f"Document: {node_with_score.node.metadata['file_name']}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# Employee Benefits
# The first chunk was very relevant because it came from the employee benefits document.
# The answer sounded confident and specific.
# I did not expect the mission and partnerships documents to also be retrieved.

# Security Policies
# The first chunk was very relevant because it came from the security policy document.
# The answer sounded confident and specific.
# I did not expect the employee benefits and mission documents to also be retrieved.


# LlamaIndex Question 2

question = "What employee benefits does BrightLeaf offer?"

query_engine_1 = index.as_query_engine(similarity_top_k=1)
response_1 = query_engine_1.query(question)

print("\nTop K = 1")
print("Response:", response_1)

for node_with_score in response_1.source_nodes:
    print(f"Document: {node_with_score.node.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")


query_engine_5 = index.as_query_engine(similarity_top_k=5)
response_5 = query_engine_5.query(question)

print("\nTop K = 5")
print("Response:", response_5)

for node_with_score in response_5.source_nodes:
    print(f"Document: {node_with_score.node.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")

# The two responses were very similar.
# The top_k=5 answer included a little more detail because it used more documents.
# More context is not always better because extra documents may not be useful to the question.

# LlamaIndex Question 3

question = "What is BrightLeaf's stock price?"

response = query_engine.query(question)

print("\nQuestion:", question)
print("Response:", response)

for node_with_score in response.source_nodes:
    print(f"Document: {node_with_score.node.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text: {node_with_score.node.get_content()}")
    print("-" * 30)

# I expected the model to say that the stock price was not in the documents.
# The model did that, but it still retrieved other BrightLeaf documents that did not contain the answer.
# This happened because the system still looked for the closest matches even though none answered the question.
# I would add a minimum similarity score so the system can reject results that are not relevant enough.

# LlamaIndex Question 4


# Create judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Create evaluators
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# First query
q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

faithfulness_result = faithfulness_evaluator.evaluate_response(
    query=q,
    response=response
)

relevancy_result = relevancy_evaluator.evaluate_response(
    query=q,
    response=response
)

print("\nQ4 First Query:", q)
print("Faithfulness Score:", faithfulness_result.score)
print("Relevancy Score:", relevancy_result.score)


# Second query
q2 = "What is BrightLeaf's stock price?"
response2 = query_engine.query(q2)

faithfulness_result2 = faithfulness_evaluator.evaluate_response(
    query=q2,
    response=response2
)

relevancy_result2 = relevancy_evaluator.evaluate_response(
    query=q2,
    response=response2
)

print("\nQ4 Second Query:", q2)
print("Faithfulness Score:", faithfulness_result2.score)
print("Relevancy Score:", relevancy_result2.score)

# A faithfulness score of 1.0 means the response is supported by the retrieved information.
# A score of 0.0 would mean the response includes information that is not supported.

# Relevancy checks whether the response answers the question.
# Faithfulness checks whether the response is supported by the retrieved information.

# I expected the stock price query to get lower scores, but both queries received 1.0 for both scores.
# The stock price response still scored well because the model correctly said the information was not provided.
# This shows that a question with missing information does not always produce lower evaluation scores.

# LLM-as-a-judge uses another LLM to evaluate the response.
# This is useful because written answers can be correct in different ways that a simple accuracy check may miss.

