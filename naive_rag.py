import os
import json
import glob
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from answer_generator import normalize as norm_text, detect_question_type, extract_short_answer


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def load_text_files(folder: str) -> Dict[str, str]:
    files = sorted(glob.glob(os.path.join(folder, "*.*")))
    docs: Dict[str, str] = {}
    for path in files:
        if path.endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8") as f:
                docs[os.path.basename(path)] = f.read()
    if not docs:
        raise FileNotFoundError(f"No .txt/.md files found in folder: {folder}")
    return docs


def sliding_window_chunk(text: str, chunk_size_words: int = 100, overlap_words: int = 20) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size_words - overlap_words)
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_size_words)
        chunk_words = words[start:end]
        if len(chunk_words) < max(30, chunk_size_words // 4):
            break
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
    return chunks


class NaiveRAG:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks: List[Chunk] = []
        self.dim = None

    def build_index(self, docs: Dict[str, str], chunk_size_words: int = 100, overlap_words: int = 20):
        all_chunks: List[Chunk] = []
        for doc_id, text in docs.items():
            pieces = sliding_window_chunk(text, chunk_size_words, overlap_words)
            for i, chunk_text in enumerate(pieces):
                all_chunks.append(Chunk(doc_id=doc_id, chunk_id=i, text=chunk_text))

        if not all_chunks:
            raise ValueError("No chunks created. Check your documents / chunking params.")

        self.chunks = all_chunks

        texts = [c.text for c in self.chunks]
        embs = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        embs = np.asarray(embs, dtype=np.float32)

        self.dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)  # cosine via inner product because normalized
        self.index.add(embs)

    def retrieve(self, query: str, top_k: int = 6) -> List[Tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)

        scores, ids = self.index.search(q_emb, top_k)
        results: List[Tuple[Chunk, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def _split_sentences(self, text: str) -> List[str]:
        # Split on newlines to handle headers that lack punctuation
        raw_lines = text.splitlines()
        final_sentences = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            # Split line by punctuation
            pieces = re.split(r"(?<=[.!?])\s+", line)
            for p in pieces:
                if p.strip():
                    final_sentences.append(p.strip())
        
        if not final_sentences:
            return [text.strip()]
        
        return final_sentences

    def _best_sentence(self, query: str, text: str) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return text.strip()
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        s_embs = self.embedder.encode(sentences, normalize_embeddings=True)
        q = np.asarray(q_emb, dtype=np.float32)[0]
        S = np.asarray(s_embs, dtype=np.float32)
        sims = S @ q
        best_idx = int(np.argmax(sims))
        return sentences[best_idx]

    def generate_extractive_answer(
        self,
        query: str,
        retrieved: List[Tuple[Chunk, float]],
        min_score: float = 0.0,
    ) -> Tuple[str, str]:
        if not retrieved:
            return ("", "Retrieved Context: (none)")
        chosen: Optional[Tuple[Chunk, float]] = None
        for c, s in retrieved:
            if s >= min_score:
                chosen = (c, s)
                break
        if chosen is None:
            chosen = retrieved[0]
        best_sentence = self._best_sentence(query, chosen[0].text)
        qtype = detect_question_type(query)
        answer = extract_short_answer(best_sentence, qtype, query)
        context = "\n\n".join(
            [f"[{c.doc_id}#{c.chunk_id} | score={s:.3f}]\n{c.text}" for c, s in retrieved]
        )
        report = f"Retrieved Context:\n{context}\n\nChosen sentence: {best_sentence}"
        return answer, report


def run_case_study_txt(
    data_dir: str = "data",
    out_dir: str = "runs_txt",
    top_k: int = 6,
    chunk_size_words: int = 100,
    overlap_words: int = 20,
    min_retrieval_score: float = 0.35,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load your .txt/.md corpus
    docs = load_text_files(data_dir)

    # 2) Build index
    rag = NaiveRAG()
    rag.build_index(docs, chunk_size_words=chunk_size_words, overlap_words=overlap_words)

    # 3) Your test questions (edit these freely)
    tests = [
        {
            "id": "why_bangladesh_replaced",
            "query": "Why was Bangladesh replaced in the ICC Men’s T20 World Cup 2026?",
            "expected": "Because Bangladesh refused to play matches in India due to security concerns.",
        },
        {
            "id": "who_replaced_bangladesh",
            "query": "Which team replaced Bangladesh in the tournament?",
            "expected": "Scotland.",
        },
        {
            "id": "hosts_and_start_date",
            "query": "Who is hosting the ICC Men’s T20 World Cup 2026 and when does it start?",
            "expected": "Co-hosted by India and Sri Lanka, starting February 7, 2026.",
        },
        {
            "id": "australia_women_captain",
            "query": "Who is the new captain of Australia’s women’s cricket team?",
            "expected": "Sophie Molineux.",
        },
        {
            "id": "multi_hop_group",
            "query": "Which group is the team that replaced Bangladesh in?",
            "expected": "Group C.",
        },
    ]

    # 4) Run
    for t in tests:
        retrieved = rag.retrieve(t["query"], top_k=top_k)
        filtered = [(c, s) for c, s in retrieved if s >= min_retrieval_score]
        answer, context_report = rag.generate_extractive_answer(
            t["query"], filtered if filtered else retrieved, min_score=min_retrieval_score
        )

        expected = t["expected"].strip()
        em = norm_text(answer) == norm_text(expected)
        ans_emb = rag.embedder.encode([answer], normalize_embeddings=True)
        exp_emb = rag.embedder.encode([expected], normalize_embeddings=True)
        # Robust cosine: embeddings are normalized, so dot product equals cosine.
        ans_vec = np.asarray(ans_emb, dtype=np.float32).reshape(-1)
        exp_vec = np.asarray(exp_emb, dtype=np.float32).reshape(-1)
        cosine = float(np.dot(ans_vec, exp_vec))

        record = {
            "test_id": t["id"],
            "query": t["query"],
            "expected": t["expected"],
            "top_k": top_k,
            "retrieved": [
                {
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "score": s,
                    "text": c.text,
                }
                for c, s in retrieved
            ],
            "answer": answer,
            "report": {
                "question": t["query"],
                "context": context_report,
            },
            "metrics": {
                "exact_match": em,
                "cosine_similarity": round(cosine, 3),
            },
        }

        out_path = os.path.join(out_dir, f"{t['id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"\n=== {t['id']} ===")
        print("Q:", t["query"])
        print("Expected:", t["expected"])
        print("Answer:", answer)
        print("EM:", em, " Cosine:", round(cosine, 3))
        print("Top chunks:", [(c.doc_id, c.chunk_id, round(s, 3)) for c, s in retrieved])
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_case_study_txt()
