import os
import sys
from pathlib import Path
from typing import Any, cast



# Virtual Environment Bootstrap
def bootstrap_project_venv() -> None:
    """
    Ensures the script always runs inside the project's virtual environment.
    If not already using the venv Python interpreter, it re-launches itself using it.
    """
    venv_python = Path(__file__).resolve().parent / "venv" / "Scripts" / "python.exe"
    current_executable = Path(sys.executable).resolve()

    # If venv exists and current Python is not the venv Python → restart script using venv
    if venv_python.exists() and current_executable != venv_python.resolve():
        os.execv(
            str(venv_python),
            [str(venv_python), "-u", str(Path(__file__).resolve()), *sys.argv[1:]]
        )


# Activate venv if needed
bootstrap_project_venv()



# Imports after environment setup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Load environment variables from .env file
load_dotenv()

# Directory where Chroma vector DB is stored
persist_directory = "db/chroma_db"



# Embedding Configuration
def resolve_embedding_config():
    """
    Determines which embedding provider to use (OpenAI or GitHub Models)
    and returns the model name + config.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    # Ensure API key exists
    if not api_key:
        raise EnvironmentError(
            "Missing OPENAI_API_KEY. Add it to your .env file."
        )

    config = {"api_key": api_key}
    model = os.getenv("EMBEDDING_MODEL", "").strip()

    # Detect GitHub token format
    is_github_token = (
        api_key.startswith("github_")
        or api_key.startswith("ghp_")
        or api_key.startswith("github_pat_")
    )

    if is_github_token:
        # Use GitHub Models endpoint
        config["base_url"] = os.getenv(
            "GITHUB_MODELS_BASE_URL",
            "https://models.github.ai/inference"
        )

        # Default model if not provided
        if not model:
            model = "openai/text-embedding-3-large"

        print("Using GitHub Models for embeddings.")
        return model, config

    # Validate OpenAI API key format
    if not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
        raise EnvironmentError(
            "Invalid OPENAI_API_KEY format."
        )

    # Optional custom OpenAI endpoint
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if openai_base_url:
        config["base_url"] = openai_base_url

    # Default embedding model
    if not model:
        model = "text-embedding-3-small"

    print("Using OpenAI for embeddings.")
    return model, config



# Chat Model Configuration
def resolve_chat_config():
    """
    Determines which chat model provider to use and returns config.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY.")

    config = {"api_key": api_key}
    model = os.getenv("CHAT_MODEL", "").strip()

    # Detect GitHub token
    is_github_token = (
        api_key.startswith("github_")
        or api_key.startswith("ghp_")
        or api_key.startswith("github_pat_")
    )

    if is_github_token:
        config["base_url"] = os.getenv(
            "GITHUB_MODELS_BASE_URL",
            "https://models.github.ai/inference"
        )

        if not model:
            model = "openai/gpt-4o-mini"

        print("Using GitHub Models for chat completions.")
        return model, config

    # Validate OpenAI key
    if not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
        raise EnvironmentError("Invalid OPENAI_API_KEY format.")

    # Optional custom endpoint
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if openai_base_url:
        config["base_url"] = openai_base_url

    # Default chat model
    if not model:
        model = "gpt-4o-mini"

    print("Using OpenAI for chat completions.")
    return model, config



# Initialize Embeddings + Vector DB
embedding_model_name, embedding_client_config = resolve_embedding_config()

# Create embedding model instance
embedding_model = OpenAIEmbeddings(
    model=embedding_model_name,
    chunk_size=32,  # batch size for embedding requests
    **embedding_client_config
)

# Initialize Chroma vector database
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},  # similarity metric
)

# Initialize Chat Model
chat_model_name, chat_client_config = resolve_chat_config()

model = ChatOpenAI(
    model=chat_model_name,
    **cast(Any, chat_client_config)
)


# Store conversation history
chat_history: list[HumanMessage | AIMessage] = []


# Utility Function
def content_to_text(content: Any) -> str:
    """
    Ensures model output is always converted to string.
    """
    if isinstance(content, str):
        return content
    return str(content)


# Core RAG Function
def ask_question(user_question: str):
    """
    Handles:
    1. Question rewriting (for better retrieval)
    2. Document retrieval
    3. Answer generation using retrieved context
    """
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Rewrite question if chat history exists
    if chat_history:
        messages = [
            SystemMessage(
                content="Rewrite the question to be standalone and searchable."
            ),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = content_to_text(result.content).strip()

        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Retrieve relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")

    for i, doc in enumerate(docs, 1):
        preview = "\n".join(doc.page_content.split("\n")[:2])
        print(f"  Doc {i}: {preview}...")

    # Step 3: Combine retrieved context with question
    combined_input = f'''Based on the following documents, answer the question:

Question: {user_question}

Documents:
{"\n".join([f"- {doc.page_content}" for doc in docs])}

Only use the provided documents. If insufficient, say you don't have enough information.
'''

    # Step 4: Generate answer
    messages = [
        SystemMessage(
            content="Answer based only on provided documents."
        ),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = content_to_text(result.content)

    # Step 5: Save conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer


# CLI Chat Loop
def start_chat():
    """
    Starts an interactive terminal chat session.
    """
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


# Entry Point
if __name__ == "__main__":
    start_chat()