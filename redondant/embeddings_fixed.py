"""
CONNECTIQ EMBEDDINGS PIPELINE v2.1 - PRODUCTION SAFE
=====================================================
Production-grade embedding generator for RAG retrieval.
Enhanced with comprehensive logging, validation, and error handling.

Author: ConnectIQ Team
License: Proprietary - Commercial Use
"""

import os
import json
import time
import hashlib
import sys
from datetime import datetime, timezone
from typing import List, Dict, Optional
from pathlib import Path

try:
    import openai
except ImportError:
    print("❌ ERROR: OpenAI library not installed")
    print("   Run: pip install openai")
    sys.exit(1)


class EmbeddingError(RuntimeError):
    """Custom exception for embedding pipeline failures"""
    pass


class EmbeddingPipeline:
    """
    Production-grade embedding pipeline with:
    - Comprehensive logging
    - Batch processing with progress
    - Robust error handling & retries
    - Content preservation for retrieval
    - Atomic file writes
    """

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
            "errors": 0,
            "retries": 0,
        }

        self._load_api_key()

    # ===============================================================
    # INITIALIZATION
    # ===============================================================

    def _log(self, message: str, level: str = "INFO"):
        """Centralized logging"""
        if not self.verbose and level == "DEBUG":
            return
        
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍",
            "PROGRESS": "⏳",
        }.get(level, "•")
        
        print(f"{prefix} [{level}] {message}")

    def _load_api_key(self):
        """Load and validate OpenAI API key"""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise EmbeddingError(
                "OPENAI_API_KEY environment variable not set.\n"
                "Set it with: export OPENAI_API_KEY='sk-...'"
            )
        
        if not api_key.startswith("sk-"):
            self._log("API key format looks suspicious (should start with 'sk-')", "WARNING")
        
        openai.api_key = api_key
        self._log("OpenAI API key loaded", "DEBUG")

    # ===============================================================
    # LOADING & VALIDATION
    # ===============================================================

    def load(self):
        """Load and validate RAG chunks with comprehensive checks"""
        self._log(f"Loading input file: {self.input_file}")
        
        # File existence check
        if not Path(self.input_file).exists():
            raise EmbeddingError(f"Input file not found: {self.input_file}")
        
        # File size check
        file_size = Path(self.input_file).stat().st_size
        if file_size == 0:
            raise EmbeddingError("Input file is empty (0 bytes)")
        
        self._log(f"File size: {file_size:,} bytes", "DEBUG")
        
        # Load JSON
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            raise EmbeddingError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise EmbeddingError(f"Cannot read input file: {e}")

        # Validate structure
        if not isinstance(payload, dict):
            raise EmbeddingError("Invalid input format: root is not a dict")

        if "chunks" not in payload:
            raise EmbeddingError("Invalid loader output: missing 'chunks' key")

        if not isinstance(payload["chunks"], list):
            raise EmbeddingError("'chunks' must be a list")

        if len(payload["chunks"]) == 0:
            raise EmbeddingError("Loader output contains ZERO chunks")

        self.chunks = payload["chunks"]
        self.stats["chunks_loaded"] = len(self.chunks)
        
        # Validate chunk structure
        self._validate_chunks()
        
        self._log(f"Loaded {len(self.chunks)} chunks", "SUCCESS")
        
        # Log metadata if present
        if "metadata" in payload:
            meta = payload["metadata"]
            self._log(f"Generator: {meta.get('generator', 'unknown')}", "DEBUG")
            self._log(f"Total chunks: {meta.get('total_chunks', 0)}", "DEBUG")

    def _validate_chunks(self):
        """Validate chunk structure"""
        required_fields = ["chunk_id", "content"]
        
        for idx, chunk in enumerate(self.chunks[:5]):  # Check first 5
            if not isinstance(chunk, dict):
                raise EmbeddingError(f"Chunk {idx} is not a dict")
            
            for field in required_fields:
                if field not in chunk:
                    raise EmbeddingError(f"Chunk {idx} missing required field: {field}")
            
            if not chunk["content"] or len(chunk["content"].strip()) == 0:
                self._log(f"Chunk {idx} has empty content", "WARNING")

    # ===============================================================
    # EMBEDDING
    # ===============================================================

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with retry logic.
        
        CRITICAL: Preserves order - texts[i] -> embeddings[i]
        """
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                self.stats["api_calls"] += 1
                
                response = openai.embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float",  # Explicit format
                )
                
                # Extract embeddings in correct order
                embeddings = [item.embedding for item in response.data]
                
                if len(embeddings) != len(texts):
                    raise EmbeddingError(
                        f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}"
                    )
                
                return embeddings

            except openai.RateLimitError as e:
                retries += 1
                self.stats["retries"] += 1
                wait_time = min(2 ** retries, 60)  # Max 60s
                
                self._log(
                    f"Rate limit hit, waiting {wait_time}s (retry {retries}/{self.max_retries})",
                    "WARNING"
                )
                
                if retries > self.max_retries:
                    raise EmbeddingError(f"Rate limit exceeded after {self.max_retries} retries") from e
                
                time.sleep(wait_time)
                last_error = e
            
            except openai.APIError as e:
                retries += 1
                self.stats["retries"] += 1
                self.stats["errors"] += 1
                
                self._log(f"API error: {e} (retry {retries}/{self.max_retries})", "WARNING")
                
                if retries > self.max_retries:
                    raise EmbeddingError(f"API error after {self.max_retries} retries: {e}") from e
                
                time.sleep(2 ** retries)
                last_error = e
            
            except Exception as e:
                self.stats["errors"] += 1
                raise EmbeddingError(f"Unexpected embedding error: {type(e).__name__}: {e}") from e
        
        # Should never reach here, but just in case
        raise EmbeddingError(f"Embedding failed: {last_error}")

    def _embedding_id(self, chunk_id: str) -> str:
        """Generate deterministic embedding ID"""
        raw = f"embed_{chunk_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ===============================================================
    # PROCESSING
    # ===============================================================

    def process(self):
        """Process chunks in batches with progress tracking"""
        self._log("Starting embedding generation...")
        
        if not self.chunks:
            raise EmbeddingError("No chunks to process")
        
        total = len(self.chunks)
        total_batches = (total + self.batch_size - 1) // self.batch_size
        
        start_time = time.time()
        
        for batch_idx in range(0, total, self.batch_size):
            batch = self.chunks[batch_idx : batch_idx + self.batch_size]
            batch_num = (batch_idx // self.batch_size) + 1
            
            # Extract texts (with validation)
            texts = []
            for chunk in batch:
                content = chunk.get("content", "").strip()
                if not content:
                    self._log(
                        f"Chunk {chunk.get('chunk_id', 'unknown')} has empty content, using placeholder",
                        "WARNING"
                    )
                    content = "[Empty content]"
                texts.append(content)
            
            # Generate embeddings
            self._log(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} chunks, {batch_idx + len(batch)}/{total} total)",
                "PROGRESS"
            )
            
            try:
                vectors = self._embed_batch(texts)
            except EmbeddingError as e:
                self._log(f"Failed to embed batch {batch_num}: {e}", "ERROR")
                raise
            
            # Create records (CRITICAL: preserve content for retrieval)
            for chunk, vector in zip(batch, vectors):
                record = {
                    "embedding_id": self._embedding_id(chunk["chunk_id"]),
                    "chunk_id": chunk["chunk_id"],
                    "parent_doc_id": chunk.get("parent_doc_id", ""),
                    "url": chunk.get("url", ""),
                    "page_type": chunk.get("page_type", "unknown"),
                    "title": chunk.get("title", ""),
                    "content": chunk.get("content", ""),  # CRITICAL: preserve for chatbot
                    "content_length": len(chunk.get("content", "")),
                    "vector": vector,
                    "vector_dim": len(vector),
                    "model": self.model,
                    "source": "connectiq_embeddings_v2.1",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

                self.embeddings.append(record)
                self.stats["embeddings_created"] += 1
            
            # Rate limiting
            if batch_num < total_batches:  # Don't sleep after last batch
                time.sleep(self.rate_limit_delay)
        
        elapsed = time.time() - start_time
        avg_per_chunk = elapsed / total if total > 0 else 0
        
        self._log(
            f"Processing complete: {total} chunks in {elapsed:.1f}s "
            f"({avg_per_chunk:.2f}s/chunk)",
            "SUCCESS"
        )

    # ===============================================================
    # OUTPUT
    # ===============================================================

    def save(self):
        """Save embeddings with atomic write"""
        self._log(f"Saving output to: {self.output_file}")
        
        if not self.embeddings:
            raise EmbeddingError("Cannot save: no embeddings to write")
        
        # Calculate stats
        total_vectors = sum(len(e["vector"]) for e in self.embeddings)
        avg_dim = total_vectors // len(self.embeddings) if self.embeddings else 0
        
        output = {
            "metadata": {
                "pipeline": "connectiq_embeddings_v2.1",
                "model": self.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "input_file": self.input_file,
                "total_embeddings": len(self.embeddings),
                "vector_dimension": avg_dim,
                "config": {
                    "batch_size": self.batch_size,
                    "rate_limit_delay": self.rate_limit_delay,
                    "max_retries": self.max_retries,
                },
                "stats": self.stats,
            },
            "embeddings": self.embeddings,
        }
        
        try:
            # Ensure output directory exists
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write (temp file + rename)
            temp_file = f"{self.output_file}.tmp"
            
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            # Atomic rename
            Path(temp_file).replace(output_path)
            
            file_size = output_path.stat().st_size
            self._log(f"Output file created: {file_size:,} bytes", "SUCCESS")
            
        except Exception as e:
            raise EmbeddingError(f"Failed to write output file: {e}") from e

    # ===============================================================
    # PIPELINE EXECUTION
    # ===============================================================

    def run(self):
        """Execute complete embedding pipeline"""
        try:
            print("\n" + "="*70)
            print("🧬 CONNECTIQ EMBEDDINGS PIPELINE v2.1")
            print("="*70)
            
            # Load
            self.load()
            
            # Process
            self.process()
            
            # Save
            self.save()
            
            # Summary
            self._print_summary()
            
            return True
            
        except EmbeddingError as e:
            self._log(f"Pipeline failed: {e}", "ERROR")
            self._print_summary(failed=True)
            raise
        
        except Exception as e:
            self._log(f"Unexpected error: {type(e).__name__}: {e}", "ERROR")
            self._print_summary(failed=True)
            raise EmbeddingError(f"Unexpected error: {e}") from e

    def _print_summary(self, failed: bool = False):
        """Print execution summary"""
        print("\n" + "="*70)
        
        if failed:
            print("❌ EMBEDDINGS PIPELINE FAILED")
        else:
            print("✅ EMBEDDINGS PIPELINE SUCCESS")
        
        print("="*70)
        print(f"  Input file:          {self.input_file}")
        print(f"  Output file:         {self.output_file}")
        print(f"  Model:               {self.model}")
        print(f"  Chunks loaded:       {self.stats['chunks_loaded']}")
        print(f"  Embeddings created:  {self.stats['embeddings_created']}")
        print(f"  API calls:           {self.stats['api_calls']}")
        print(f"  Retries:             {self.stats['retries']}")
        print(f"  Errors:              {self.stats['errors']}")
        
        if self.embeddings:
            vector_dim = len(self.embeddings[0]["vector"])
            total_size = sum(len(e["content"]) for e in self.embeddings)
            avg_content = total_size // len(self.embeddings)
            print(f"  Vector dimension:    {vector_dim}")
            print(f"  Avg content length:  {avg_content} chars")
        
        print("="*70 + "\n")


# ===============================================================
# CLI
# ===============================================================

def main():
    """CLI entry point with improved argument parsing"""
    
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║       CONNECTIQ EMBEDDINGS PIPELINE v2.1 - PRODUCTION SAFE       ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  python embeddings_fixed.py <input_file> [output_file] [options]

Arguments:
  input_file         RAG chunks JSON from loader (required)
  output_file        Output embeddings JSON (default: embeddings.json)

Options:
  --model NAME           OpenAI model (default: text-embedding-3-large)
  --batch-size N         Batch size (default: 50)
  --rate-limit N         Delay between batches in seconds (default: 0.5)
  --max-retries N        Max retry attempts (default: 3)
  --quiet                Suppress debug output
  -h, --help             Show this help

Environment:
  OPENAI_API_KEY         Required - your OpenAI API key

Examples:
  python embeddings_fixed.py rag_documents.json
  python embeddings_fixed.py rag_documents.json embeddings.json
  python embeddings_fixed.py data.json out.json --batch-size 100 --rate-limit 1.0
  python embeddings_fixed.py rag_documents.json --model text-embedding-3-small --quiet
        """)
        sys.exit(0)

    # Parse required arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "embeddings.json"
    
    # Parse optional arguments
    kwargs = {
        "verbose": True,
    }
    
    i = 3 if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == "--model" and i + 1 < len(sys.argv):
            kwargs["model"] = sys.argv[i + 1]
            i += 2
        elif arg == "--batch-size" and i + 1 < len(sys.argv):
            kwargs["batch_size"] = int(sys.argv[i + 1])
            i += 2
        elif arg == "--rate-limit" and i + 1 < len(sys.argv):
            kwargs["rate_limit_delay"] = float(sys.argv[i + 1])
            i += 2
        elif arg == "--max-retries" and i + 1 < len(sys.argv):
            kwargs["max_retries"] = int(sys.argv[i + 1])
            i += 2
        elif arg == "--quiet":
            kwargs["verbose"] = False
            i += 1
        else:
            print(f"❌ Unknown option: {arg}")
            print("Use --help for usage information")
            sys.exit(1)
    
    # Execute pipeline
    try:
        pipeline = EmbeddingPipeline(input_file, output_file, **kwargs)
        pipeline.run()
        sys.exit(0)
        
    except EmbeddingError as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        sys.exit(2)
    
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()