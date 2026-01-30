import json
import numpy as np
from naive_rag import NaiveRAG, load_text_files, norm_text

def evaluate_system():
    # 1. Setup
    print("🚀 Initializing System...")
    docs = load_text_files("data")
    rag = NaiveRAG()
    rag.build_index(docs, chunk_size_words=100, overlap_words=20)
    
    with open("data/golden_dataset.json", "r") as f:
        dataset = json.load(f)

    # 2. Metrics Containers
    total_queries = len(dataset)
    retrieval_hits = 0
    answer_exact_matches = 0
    
    print(f"\n📊 Starting Evaluation on {total_queries} Test Cases...\n")
    print(f"{ 'ID':<20} | {'Hit?':<5} | {'Match?':<5} | {'Query':<40}")
    print("-" * 80)

    # 3. Run Loop
    for item in dataset:
        query = item["query"]
        gold_substring = item["gold_chunk_substring"]
        expected_answer = item["expected_answer"]
        
        # Run RAG
        retrieved_chunks = rag.retrieve(query, top_k=3)
        answer, _ = rag.generate_extractive_answer(query, retrieved_chunks)
        
        # --- Metric 1: Retrieval Hit Rate ---
        # Did any retrieved chunk contain the unique "gold" substring?
        is_hit = False
        for chunk, score in retrieved_chunks:
            if gold_substring in chunk.text:
                is_hit = True
                break
        
        if is_hit:
            retrieval_hits += 1
            
        # --- Metric 2: Generation Accuracy (Exact Match) ---
        # Normalize both to ignore capitalization/punctuation
        is_match = norm_text(answer) == norm_text(expected_answer)
        if is_match:
            answer_exact_matches += 1
            
        hit_icon = "✅" if is_hit else "❌"
        match_icon = "✅" if is_match else "❌"
        print(f"{item['id']:<20} | {hit_icon:<5} | {match_icon:<5} | {query[:38]}...")

    # 4. Final Report
    print("\n" + "="*40)
    print("📉 FINAL RESULTS")
    print("="*40)
    print(f"Retrieval Hit Rate @ 3:  {retrieval_hits/total_queries:.2%}")
    print(f"Answer Exact Match:      {answer_exact_matches/total_queries:.2%}")
    print("="*40)
    
    if retrieval_hits < total_queries:
        print("\n🔍 CONCLUSION: Retrieval Failed.")
        print("This proves 'Lost in the Middle' or 'Embedding Drift'.")
    
    if answer_exact_matches < retrieval_hits:
        print("\n🧠 CONCLUSION: Reasoning Failed.")
        print("The context was found, but the answer was wrong.")
        print("This proves 'Extraction Errors' or 'Reasoning Gaps'.")

if __name__ == "__main__":
    evaluate_system()
