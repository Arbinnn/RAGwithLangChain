import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

persist_directory = "db/chroma_db"
def resolve_embedding_config():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "Missing OPENAI_API_KEY. Add it to your .env file as OPENAI_API_KEY=your_key "
            "or set it in your shell before running the pipeline."
        )

    config = {"api_key": api_key}
    model = os.getenv("EMBEDDING_MODEL", "").strip()

    is_github_token = (
        api_key.startswith("github_")
        or api_key.startswith("ghp_")
        or api_key.startswith("github_pat_")
    )

    if is_github_token:
        config["base_url"] = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference")
        if not model:
            model = "openai/text-embedding-3-large"
        print("Using GitHub Models for embeddings.")
        return model, config

    if not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
        raise EnvironmentError(
            "OPENAI_API_KEY format looks invalid. It should usually start with 'sk-' or 'sk-proj-'."
        )

    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if openai_base_url:
        config["base_url"] = openai_base_url

    if not model:
        model = "text-embedding-3-small"

    print("Using OpenAI for embeddings.")
    return model, config


def resolve_chat_config():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "Missing OPENAI_API_KEY. Add it to your .env file as OPENAI_API_KEY=your_key "
            "or set it in your shell before running the pipeline."
        )

    config = {"api_key": api_key}
    model = os.getenv("CHAT_MODEL", "").strip()

    is_github_token = (
        api_key.startswith("github_")
        or api_key.startswith("ghp_")
        or api_key.startswith("github_pat_")
    )

    if is_github_token:
        config["base_url"] = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference")
        if not model:
            model = "openai/gpt-4o-mini"
        print("Using GitHub Models for chat completions.")
        return model, config

    if not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
        raise EnvironmentError(
            "OPENAI_API_KEY format looks invalid. It should usually start with 'sk-' or 'sk-proj-'."
        )

    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if openai_base_url:
        config["base_url"] = openai_base_url

    if not model:
        model = "gpt-4o-mini"

    print("Using OpenAI for chat completions.")
    return model, config

#load embeddings and vector store
model, client_config = resolve_embedding_config()
embedding_model = OpenAIEmbeddings(model=model, chunk_size=32, **client_config) # type: ignore

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

#search for relevant documents
query = "What was NVIDIA's first graphics accelerator called?"

retriever= db.as_retriever(search_kwargs={"k": 5}) # k is the number of relevant documents to retrieve

relevant_docs = retriever.invoke(query) #invoke the retriever with the query to get relevant documents

print(f"Query: {query}\n")
print("Relevant Documents:")
for i, doc in enumerate(relevant_docs):
    print(f"\nDocument {i+1}")
    print(f"Source: {doc.metadata['source']}") # type: ignore
    print(f"Content: {doc.page_content} ")

#Combine the query and the relevant document contents
combined_input = f""" Based on the following documents, Please answer this questions: {query}

Documents:
{chr(10).join([doc.page_content for doc in relevant_docs])}

Please provide a concise answer based on the information from the documents. If the answer is not found in the documents, please say "Answer not found in the provided documents."""

#Create a chatOpenAI model
chat_model_name, chat_client_config = resolve_chat_config()
model = ChatOpenAI(model=chat_model_name, **chat_client_config)

#Define the message for the chat model
messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
    HumanMessage(content=combined_input)
]

#Invoke the model with the combined input
result=model.invoke(messages)

print(f"\nAnswer: {result.content}")

# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"