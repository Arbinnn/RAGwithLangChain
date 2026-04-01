import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


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

def load_documents(docs_path="docs"): # type: ignore
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The specified path '{docs_path}' does not exist.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader, # Use TextLoader for .txt files
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
    )
    documents = loader.load() # Load documents from the specified directory

    if len(documents) == 0:
        raise ValueError(f"No documents found in the specified path '{docs_path}'. Please check the directory and try again.")
    
    # for i, doc in enumerate(documents[:2]): #show first 2 documents
    #     print(f"\nDocument {i+1}")
    #     print(f"Source: {doc.metadata['source']}") # type: ignore
    #     print(f"Content Length: {len(doc.page_content)} characters")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=150):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents) # Split documents into chunks
    
    # if chunks:

    #     for i, chunk in enumerate(chunks[:5]): #show first 5 chunks
    #         print(f"\nChunk {i+1}")
    #         print(f"Source: {chunk.metadata['source']}") # type: ignore
    #         print(f"Content Length: {len(chunk.page_content)} characters") # type: ignore
    #         print(f"Metadata: {chunk.metadata}") # type: ignore
    #         print(f"Content Preview: {chunk.page_content[:200]}...") # type: ignore
    #         print("-" * 50)

    #     if len(chunks) > 5:
    #         print(f"\n...and {len(chunks) - 5} more chunks.")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    model, client_config = resolve_embedding_config()
    embedding_model = OpenAIEmbeddings(model=model, chunk_size=32, **client_config) # type: ignore

    print(f"Creating vector store with {len(chunks)} chunks...") # type: ignore
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Vector store created and persisted at '{persist_directory}'.")
    return vector_store

def main():
    print("Starting the ingestion pipeline")

    #Load documents
    documents = load_documents(docs_path="docs") # pyright: ignore[reportUnusedVariable]

    # Split loaded docs into chunks
    chunks = split_documents(documents) # pyright: ignore[reportUnusedVariable]

    # Build and persist vector store
    vector_store = create_vector_store(chunks) # pyright: ignore[reportUnusedVariable]

    print("Ingestion pipeline completed successfully.")


if __name__ == "__main__":
    main()