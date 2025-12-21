"""
CONNECTIQ AI LOADER – PRODUCTION SAFE
====================================
Strict, fail-fast loader for RAG ingestion.
If this loader succeeds, downstream pipelines are safe.
If something is wrong, it FAILS HARD.

Author: ConnectIQ Team
License: Proprietary
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import List, Dict


class LoaderError(RuntimeError):
    pass


class AILoader:
    def __init__(
        self,
        input_file: str,
        output_file: str = "rag_documents.json",
        min_doc_chars: int = 200,
        min_chunk_chars: int = 400,
        max_chunk_chars: int = 900,
        overlap_chars: int = 120,
    ):
        self.input_file = input_file
        self.output_file = output_file

        self.min_doc_chars = min_doc_chars
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

        self.documents: List[Dict] = []
        self.chunks: List[Dict] = []

    # -------------------------------------------------
    # LOAD & VALIDATE
    # -------------------------------------------------

    def load(self):
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            raise LoaderError(f"Cannot read input file: {e}")

        if not isinstance(payload, dict):
            raise LoaderError("Invalid input format: root is not a dict")

        if "documents" not in payload:
            raise LoaderError("Invalid crawler output: missing 'documents' key")

        if not isinstance(payload["documents"], list):
            raise LoaderError("'documents' must be a list")

        if len(payload["documents"]) == 0:
            raise LoaderError("Crawler output contains ZERO documents")

        self.documents = payload["documents"]

    # -------------------------------------------------
    # CHUNKING
    # -------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        text = text.strip()
        pos = 0
        length = len(text)

        while pos < length:
            end = min(pos + self.max_chunk_chars, length)
            chunk = text[pos:end]

            if len(chunk) >= self.min_chunk_chars:
                chunks.append(chunk)

            pos = end - self.overlap_chars
            if pos < 0:
                pos = 0

        return chunks

    def _chunk_id(self, base_id: str, idx: int) -> str:
        raw = f"{base_id}:{idx}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # -------------------------------------------------
    # PROCESS
    # -------------------------------------------------

    def process(self):
        for doc in self.documents:
            content = doc.get("content", "")
            if len(content) < self.min_doc_chars:
                continue

            base_id = doc.get("doc_id")
            if not base_id:
                raise LoaderError("Document missing doc_id")

            chunks = self._chunk_text(content)

            for i, chunk in enumerate(chunks):
                self.chunks.append({
                    "chunk_id": self._chunk_id(base_id, i),
                    "parent_doc_id": base_id,
                    "url": doc.get("url"),
                    "page_type": doc.get("page_type"),
                    "title": doc.get("title", ""),
                    "content": chunk,
                    "content_length": len(chunk),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })

        if len(self.chunks) == 0:
            raise LoaderError("ZERO chunks produced – pipeline aborted")

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------

    def save(self):
        output = {
            "metadata": {
                "generator": "connectiq_loader",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_chunks": len(self.chunks),
            },
            "chunks": self.chunks,
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------
    # RUN
    # -------------------------------------------------

    def run(self):
        self.load()
        self.process()
        self.save()

        print("\n✅ LOADER SUCCESS")
        print(f"   Input docs: {len(self.documents)}")
        print(f"   Chunks: {len(self.chunks)}")
        print(f"   Output: {self.output_file}")


# -------------------------------------------------
# CLI
# -------------------------------------------------

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python loader.py crawler_output.json [rag_documents.json]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "rag_documents.json"

    try:
        loader = AILoader(input_file, output_file)
        loader.run()
    except LoaderError as e:
        print(f"\n❌ LOADER FAILED: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
