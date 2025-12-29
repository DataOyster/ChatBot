"""
CONNECTIQ EMBEDDINGS PIPELINE v2.2 – PRODUCTION READY
====================================================
Enterprise-grade embeddings generator for RAG systems.

- Loads API key from .env automatically
- Validates loader output strictly
- Fails fast on zero-chunk edge cases
- Batch-safe, retry-safe, deterministic IDs
- Preserves content for retrieval
"""

import os
import json
import time
import hashlib
import sys
from datetime import datetime, timezone
from typing import List, Dict
from pathlib import Path

from dotenv import load_dotenv
import openai


# ===============================================================
# ERRORS
# ===============================================================

class EmbeddingError(RuntimeError):
    pass


# ===============================================================
# PIPELINE
# ===============================================================

class EmbeddingPipeline:
    def __init__(
        self,
        input_file: str,
        output_file: str = "embeddings.json",
        model: str = "text-embedding-3-large",
        batch_size: int = 50,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
        verbose: bool = True,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.model = model
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.verbose = verbose

        self.chunks: List[Dict] = []
        self.embeddings: List[Dict] = []

        self.stats = {
            "chunks_loaded": 0,
            "embeddings_created": 0,
            "api_calls": 0,
            "retries": 0,
            "errors": 0,
        }

        self._load_env_and_key()

    # ===========================================================
    # ENV / LOGGING
    # ===========================================================

    def _log(self, msg: str, level: str = "INFO"):
        if not self.verbose and level == "DEBUG":
            return
        prefix = {
            "INFO": "ℹ️",
            "OK": "✅",
            "WARN": "⚠️",
            "ERR": "❌",
            "PROGRESS": "⏳",
        }.get(level, "•")
        print(f"{prefix} [{level}] {msg}")

    def _load_env_and_key(self):
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingError(
                "OPENAI_API_KEY not found.\n"
                "➡ Ensure .env exists and contains:\n"
                "   OPENAI_API_KEY=sk-xxxx"
            )

        openai.api_key = api_key
        self._log("OpenAI API key loaded", "DEBUG")

    # ===========================================================
    # LOAD & VALIDATE INPUT
    # ===========================================================

    def load(self):
        if not Path(self.input_file).exists():
            raise EmbeddingError(f"Input file not found: {self.input_file}")

        with open(self.input_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if "chunks" not in payload or not isinstance(payload["chunks"], list):
            raise EmbeddingError("Invalid loader output: missing 'chunks' array")

        if len(payload["chunks"]) == 0:
            raise EmbeddingError("Loader produced ZERO chunks – aborting")

        self.chunks = payload["chunks"]
        self.stats["chunks_loaded"] = len(self.chunks)

        for i, c in enumerate(self.chunks[:5]):
            if "chunk_id" not in c or "content" not in c:
                raise EmbeddingError(f"Chunk {i} missing required fields")

        self._log(f"Loaded {len(self.chunks)} chunks", "OK")

    # ===========================================================
    # EMBEDDING
    # ===========================================================

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        retries = 0
        while retries <= self.max_retries:
            try:
                self.stats["api_calls"] += 1
                res = openai.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                return [d.embedding for d in res.data]

            except Exception as e:
                retries += 1
                self.stats["retries"] += 1
                self.stats["errors"] += 1
                if retries > self.max_retries:
                    raise EmbeddingError(f"Embedding failed after retries: {e}")
                time.sleep(2 ** retries)

    def _embed_id(self, chunk_id: str) -> str:
        return hashlib.sha256(chunk_id.encode()).hexdigest()[:16]

    def process(self):
        total = len(self.chunks)
        self._log("Starting embedding generation", "INFO")

        for i in range(0, total, self.batch_size):
            batch = self.chunks[i:i + self.batch_size]
            texts = [c["content"] or "[EMPTY]" for c in batch]

            self._log(
                f"Batch {(i//self.batch_size)+1} "
                f"({i+len(batch)}/{total})",
                "PROGRESS"
            )

            vectors = self._embed_batch(texts)

            for chunk, vec in zip(batch, vectors):
                self.embeddings.append({
                    "embedding_id": self._embed_id(chunk["chunk_id"]),
                    "chunk_id": chunk["chunk_id"],
                    "parent_doc_id": chunk.get("parent_doc_id", ""),
                    "url": chunk.get("url", ""),
                    "page_type": chunk.get("page_type", "unknown"),
                    "title": chunk.get("title", ""),
                    "content": chunk["content"],
                    "vector": vec,
                    "vector_dim": len(vec),
                    "model": self.model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
                self.stats["embeddings_created"] += 1

            time.sleep(self.rate_limit_delay)

        self._log("Embedding generation completed", "OK")

    # ===========================================================
    # SAVE
    # ===========================================================

    def save(self):
        if not self.embeddings:
            raise EmbeddingError("No embeddings generated – nothing to save")

        output = {
            "metadata": {
                "pipeline": "connectiq_embeddings_v2.2",
                "model": self.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "input_file": self.input_file,
                "stats": self.stats,
            },
            "embeddings": self.embeddings,
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        self._log(f"Saved {len(self.embeddings)} embeddings → {self.output_file}", "OK")

    # ===========================================================
    # RUN
    # ===========================================================

    def run(self):
        print("\n" + "="*70)
        print("🧬 CONNECTIQ EMBEDDINGS PIPELINE v2.2")
        print("="*70)

        self.load()
        self.process()
        self.save()

        print("="*70)
        print("✅ EMBEDDINGS PIPELINE SUCCESS")
        print("="*70)


# ===============================================================
# CLI
# ===============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python embeddings_fixed.py <rag_documents.json> [output.json]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "embeddings.json"

    pipeline = EmbeddingPipeline(
        input_file=input_file,
        output_file=output_file,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
