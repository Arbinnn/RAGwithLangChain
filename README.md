# RAG with LangChain

A Retrieval-Augmented Generation (RAG) project that uses local text files, LangChain, and Chroma.

## Overview

The project has four Python entry points:

- `ingestion_pipeline.py` builds the vector database from the documents in `docs/`.
- `retrieval_pipeline.py` loads the saved vector database and answers a single retrieval query.
- `history_aware_generation.py` runs an interactive chat loop that rewrites follow-up questions using chat history, retrieves supporting chunks, and generates answers from the retrieved context.
- `retreival_methods.py` is a retrieval playground script for comparing retrieval strategies and tuning search parameters.

## What each file does

### `ingestion_pipeline.py`

This script prepares the knowledge base.

- Loads all `.txt` files from `docs/`
- Splits documents into chunks with `RecursiveCharacterTextSplitter`
- Creates embeddings with `OpenAIEmbeddings`
- Stores the chunks in Chroma at `db/chroma_db`

Use this script first whenever the source documents change.

### `retrieval_pipeline.py`

This script is a simple one-shot retrieval demo.

- Opens the persisted Chroma database from `db/chroma_db`
- Embeds the query and retrieves the most relevant chunks
- Sends the query and retrieved text to a chat model
- Prints the final answer

It is useful for testing retrieval quality and verifying the vector database.

### `history_aware_generation.py`

This script adds a conversational layer on top of retrieval.

- Starts an interactive terminal chat session
- Rewrites follow-up questions into standalone search queries using the previous chat history
- Retrieves the most relevant chunks from Chroma
- Builds a response prompt from the retrieved documents
- Uses the chat model to answer with context from the documents

This is the most complete user-facing RAG flow in the repository.

### `retreival_methods.py`

This script is a learning sandbox focused on retrieval behavior.

- Loads the persisted Chroma database from `db/chroma_db`
- Runs retrieval in a configurable way (for example: plain similarity, score-threshold retrieval, and MMR)
- Prints retrieved chunks so you can inspect relevance and diversity directly
- Helps you tune `k`, `fetch_k`, `lambda_mult`, and score threshold values while learning RAG retrieval trade-offs

Use this script when you want to experiment with retrieval methods before wiring them into your full RAG pipeline.

## Models and tools used

- Embeddings provider:
  - OpenAI API key (`OPENAI_API_KEY` starts with `sk-` or `sk-proj-`)
  - GitHub Models token (`OPENAI_API_KEY` starts with `github_`, `ghp_`, or `github_pat_`)
- Default embedding models:
  - OpenAI path: `text-embedding-3-small`
  - GitHub Models path: `openai/text-embedding-3-large`
- Default chat models:
  - OpenAI path: `gpt-4o-mini`
  - GitHub Models path: `openai/gpt-4o-mini`
- Vector database: Chroma
- Framework: LangChain

## Project structure

- `ingestion_pipeline.py`: build and persist the vector database from the documents
- `retrieval_pipeline.py`: run a one-shot semantic retrieval query against the saved database
- `history_aware_generation.py`: run an interactive retrieval-augmented chat session with history
- `retreival_methods.py`: explore and compare retrieval methods such as similarity search and MMR
- `docs/`: source text files
- `db/chroma_db/`: persisted vector store

## Setup

1. Create and activate a virtual environment
   - Windows PowerShell:
     - `python -m venv venv`
     - `.\venv\Scripts\Activate.ps1`

2. Install dependencies
   - `pip install langchain-community langchain-text-splitters langchain-openai langchain-chroma python-dotenv`

3. Configure environment variables
   - Copy `.env.example` to `.env`
   - Set `OPENAI_API_KEY`
   - Optional: set `EMBEDDING_MODEL`
   - Optional: set `CHAT_MODEL`
   - Optional: set `OPENAI_BASE_URL` if you use a custom OpenAI-compatible endpoint
   - Optional: set `GITHUB_MODELS_BASE_URL` if you use GitHub Models

Example `.env` values:

```env
OPENAI_API_KEY=your_key_here
EMBEDDING_MODEL=openai/text-embedding-3-large
CHAT_MODEL=openai/gpt-4o-mini
GITHUB_MODELS_BASE_URL=https://models.github.ai/inference
```

## Run

1. Build the vector store
   - `python ingestion_pipeline.py`

2. Test one-shot retrieval
   - `python retrieval_pipeline.py`

3. Explore retrieval methods
   - `python retreival_methods.py`

4. Start the history-aware chat session
   - `python history_aware_generation.py`

## Notes

- Keep the same provider and embedding model between ingestion and retrieval to avoid embedding mismatch.
- If you use a GitHub token with GitHub Models, keep `GITHUB_MODELS_BASE_URL` set.
- `history_aware_generation.py` will re-launch itself with the project virtual environment on Windows if it is started from the wrong interpreter.
