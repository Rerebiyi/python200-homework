from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# --- Step 1: Setup ---

from dotenv import load_dotenv
from pathlib import Path
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path(
    "../../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs"
)

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

print("Document directory found.")

# --- Step 2: Load the Documents ---

documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"Documents loaded: {len(documents)}")

for document in documents:
    print(document.metadata["file_name"])

# --- Step 3: Build the Index and Query Engine ---

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)

print("Index built successfully. Ready to answer questions.")


# --- Step 4: Query the Assistant ---

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    response = query_engine.query(question)

    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")

    top_node = response.source_nodes[0]

    print(f"Document: {top_node.node.metadata['file_name']}")
    print(f"Similarity Score: {top_node.score:.4f}")
    print(f"Text: {top_node.node.get_content()[:200]}...")

# The assistant sounded confident and the answers were accurate.
# I was surprised that our_story.txt was the top result for the hours question.
# The other answers matched the information in the documents.

# --- Step 5: Find a Failure ---

failure_question = "How much money does Groundwork Coffee make each year?"

failure_response = query_engine.query(failure_question)

print(f"\nQuestion: {failure_question}")
print(f"Response: {failure_response}")

for node_with_score in failure_response.source_nodes:
    print(f"Document: {node_with_score.node.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text: {node_with_score.node.get_content()[:200]}...")
    print("-" * 30)

# I asked how much money Groundwork makes each year because that information is not in the documents.
# The system found related documents, but none of them had the answer.
# The model did not guess. It clearly said that the information was not provided.
# This shows that AI responses should still be checked because they may not always have the right information.
# I would make the system ignore results that are not relevant enough.


# --- Step 6: Reflection ---

# The LlamaIndex version only took about 3 lines to load the documents,
# build the index, and create the query engine.
# This shows that a framework can make RAG much easier to build.

# Another use case could be a company assistant that answers employee questions
# using documents about benefits, policies, and company rules.

# RAG cannot fully stop the model from giving a wrong answer.
# Even with the right information retrieved, the model can misunderstand it.



