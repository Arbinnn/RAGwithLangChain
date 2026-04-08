from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from typing import List
from collections import defaultdict

load_dotenv()

#setup
persistent_directory = "db/chroma_db"

def resolve_embedding_config():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "Missing OPENAI_API_KEY. Add it to your .env file as OPENAI_API_KEY=your_key."
        )

    config = {"api_key": api_key}
    model = os.getenv("EMBEDDING_MODEL", "").strip()

    is_github_token = (
        api_key.startswith("github_")
        or api_key.startswith("ghp_")
        or api_key.startswith("github_pat_")
    )

    if is_github_token:
        config["base_url"] = os.getenv(
            "GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference"
        )
        if not model:
            model = "openai/text-embedding-3-large"
        print("Using GitHub Models for embeddings.")
        return model, config

    if not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
        raise EnvironmentError(
            "OPENAI_API_KEY format looks invalid. Use an OpenAI key (sk-...) or a GitHub token (github_pat_...)."
        )

    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if openai_base_url:
        config["base_url"] = openai_base_url

    if not model:
        model = "text-embedding-3-large"

    print("Using OpenAI for embeddings.")
    return model, config


def resolve_chat_config():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "Missing OPENAI_API_KEY. Add it to your .env file as OPENAI_API_KEY=your_key."
        )

    config = {"api_key": api_key}
    model = os.getenv("CHAT_MODEL", "").strip()

    is_github_token = (
        api_key.startswith("github_")
        or api_key.startswith("ghp_")
        or api_key.startswith("github_pat_")
    )

    if is_github_token:
        config["base_url"] = os.getenv(
            "GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference"
        )
        if not model:
            model = "openai/gpt-4o-mini"
        print("Using GitHub Models for chat completions.")
        return model, config

    if not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
        raise EnvironmentError(
            "OPENAI_API_KEY format looks invalid. Use an OpenAI key (sk-...) or a GitHub token (github_pat_...)."
        )

    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if openai_base_url:
        config["base_url"] = openai_base_url

    if not model:
        model = "gpt-4o-mini"

    print("Using OpenAI for chat completions.")
    return model, config


model, client_config = resolve_embedding_config()
embedding_model = OpenAIEmbeddings(model=model, **client_config)
chat_model, chat_client_config = resolve_chat_config()
llm = ChatOpenAI(model=chat_model, temperature=0.2, **chat_client_config)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

#pydantic model for structured output
class QueryVariations(BaseModel):
    query: List[str]

#MAIN FUNCTION

#Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}")

#Generate query variations using LLM
llm_with_tools = llm.with_structured_output(QueryVariations)
prompt = f"Generate 3 different variations of the following query: '{original_query}'"

response = llm_with_tools.invoke(prompt)
query_variations = response.query

print("\nGenerated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "="*60)


#Step 2: Search with each variation and collect results

retriever= db.as_retriever(search_kwargs={"k": 5})
all_results = []

for i, query in enumerate(query_variations, 1):
    print(f"\nSearching with Query Variation {i}: '{query}'")
    results = retriever.invoke(query)
    all_results.append(results)
    print(f"Retrieved {len(results)} documents for variation {i}.")

    for j, doc in enumerate(results, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")

#Step 3: Implement Reciprocal Rank Fusion (RRF) to combine results from all variations

def reciprocal_rank_fusion(all_results, k=60, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")

    rrf_scores = defaultdict(float)
    all_unique_results = {}
    chunk_id_map = {}
    chunk_counter = 1

    # Go through each set of results and calculate RRF scores.
    for query_idx, results in enumerate(all_results):
        if verbose:
            print(f"Processing results from Query Variation {query_idx + 1}...")

        for rank, doc in enumerate(results, 1):
            # Use chunk content as the deduplication key.
            chunk_content = doc.page_content

            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"chunk_{chunk_counter}"
                chunk_counter += 1

            chunk_id = chunk_id_map[chunk_content]
            all_unique_results[chunk_id] = doc
            increment = 1 / (rank + k)
            rrf_scores[chunk_id] += increment

            if verbose:
                print(
                    f"  Position {rank}: {chunk_id} +{increment:.4f} "
                    f"(running total: {rrf_scores[chunk_id]:.4f})"
                )
                print(f"    Preview: {chunk_content[:80]}...")

    if verbose:
        print()

    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_results[chunk_id], score) for chunk_id, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(all_results)} queries.")
    
    return sorted_chunks

# Apply RRF to our retrieval results
fused_results = reciprocal_rank_fusion(all_results, k=60, verbose=True)


print("\n" + "="*60)
print("FINAL RRF RANKING")
print("="*60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f"🏆 RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(f"\n✅ RRF Complete! Fused {len(fused_results)} unique documents from {len(query_variations)} query variations.")
print("\n💡 Key benefits:")
print("   • Documents appearing in multiple queries get boosted scores")
print("   • Higher positions contribute more to the final score") 
print("   • Balanced fusion using k=60 for gentle position penalties")

