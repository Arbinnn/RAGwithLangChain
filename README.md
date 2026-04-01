# RAG with LangChain

A simple Retrieval-Augmented Generation (RAG) project using local text files, LangChain, and Chroma.

## What this project does

- Loads .txt documents from docs/
- Splits documents into chunks
- Creates embeddings and stores them in Chroma (db/chroma_db)
- Retrieves top matching chunks for a query

## Models and tools used

- Embeddings provider:
  - OpenAI API key (OPENAI_API_KEY starts with sk- or sk-proj-)
  - GitHub Models token (OPENAI*API_KEY starts with github*, ghp*, or github_pat*)
- Default embedding models:
  - OpenAI path: text-embedding-3-small
  - GitHub Models path: openai/text-embedding-3-large
- Vector database: Chroma
- Framework: LangChain

## Project structure

- ingestion_pipeline.py: Build and persist vector DB from docs
- retrieval_pipeline.py: Run semantic retrieval from saved DB
- docs/: Source text files
- db/chroma_db/: Persisted vector store
- .env.example: Environment variable template

## Setup

1. Create and activate virtual environment
   - Windows PowerShell:
     - python -m venv venv
     - .\venv\Scripts\Activate.ps1

2. Install dependencies
   - pip install langchain-community langchain-text-splitters langchain-openai langchain-chroma python-dotenv

3. Configure environment variables
   - Copy .env.example to .env
   - Set OPENAI_API_KEY
   - Optional: set EMBEDDING_MODEL

Example .env values:
OPENAI_API_KEY=your_key_here
EMBEDDING_MODEL=openai/text-embedding-3-large
GITHUB_MODELS_BASE_URL=https://models.github.ai/inference

## Run

1. Build vector store
   - python ingestion_pipeline.py

2. Test retrieval
   - python retrieval_pipeline.py

## Notes

- Keep the same provider/model between ingestion and retrieval to avoid embedding mismatch.
- If you use GitHub token + GitHub Models, keep GITHUB_MODELS_BASE_URL set.
