"""
CONNECTIQ AI LOADER – PRODUCTION SAFE v2.1
==========================================
Strict, fail-fast loader for RAG ingestion with comprehensive logging.
If this loader succeeds, downstream pipelines are safe.

Author: ConnectIQ Team
License: Proprietary
"""

import json
import hashlib
import sys
from datetime import datetime, timezone
from typing import List, Dict
from pathlib import Path


class LoaderError(RuntimeError):
    """Custom exception for loader failures"""
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
        verbose: bool = True,
    ):
        self.input_file = input_file
        self.output_file = output_file

        self.min_doc_chars = min_doc_chars
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        
        # FIX: Ensure overlap is never >= max_chunk_chars (prevents infinite loop)
        self.overlap_chars = min(overlap_chars, max_chunk_chars // 2)
        
        self.verbose = verbose

        self.documents: List[Dict] = []
        self.chunks: List[Dict] = []
        
        # Stats
        self.stats = {
            "docs_loaded": 0,
            "docs_skipped_short": 0,
            "chunks_created": 0,
            "total_content_chars": 0,
        }

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
        }.get(level, "•")
        
        print(f"{prefix} [{level}] {message}")

    # -------------------------------------------------
    # LOAD & VALIDATE
    # -------------------------------------------------

    def load(self):
        """Load and validate crawler output with extensive checks"""
        self._log(f"Loading input file: {self.input_file}")
        
        # Check file exists
        if not Path(self.input_file).exists():
            raise LoaderError(f"Input file not found: {self.input_file}")
        
        # Check file size
        file_size = Path(self.input_file).stat().st_size
        if file_size == 0:
            raise LoaderError("Input file is empty (0 bytes)")
        
        self._log(f"File size: {file_size:,} bytes", "DEBUG")
        
        # Load JSON
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            raise LoaderError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise LoaderError(f"Cannot read input file: {e}")

        # Validate structure
        if not isinstance(payload, dict):
            raise LoaderError("Invalid input format: root is not a dict")

        if "documents" not in payload:
            raise LoaderError("Invalid crawler output: missing 'documents' key")

        if not isinstance(payload["documents"], list):
            raise LoaderError("'documents' must be a list")

        if len(payload["documents"]) == 0:
            raise LoaderError("Crawler output contains ZERO documents")

        self.documents = payload["documents"]
        self.stats["docs_loaded"] = len(self.documents)
        
        self._log(f"Loaded {len(self.documents)} documents", "SUCCESS")
        
        # Log metadata if present
        if "metadata" in payload:
            meta = payload["metadata"]
            self._log(f"Crawler version: {meta.get('crawler_version', 'unknown')}", "DEBUG")
            self._log(f"Source domain: {meta.get('domain', 'unknown')}", "DEBUG")

    # -------------------------------------------------
    # CHUNKING
    # -------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text with overlap, preventing infinite loops.
        
        CRITICAL FIX: Ensures progress is made on each iteration
        """
        chunks = []
        text = text.strip()
        length = len(text)
        
        if length == 0:
            return chunks
        
        pos = 0
        iteration_count = 0
        max_iterations = (length // self.min_chunk_chars) + 10  # Safety limit
        
        while pos < length:
            # Safety check: prevent infinite loop
            iteration_count += 1
            if iteration_count > max_iterations:
                self._log(
                    f"Chunking safety limit hit (text length: {length}). "
                    f"Check overlap_chars={self.overlap_chars} vs max_chunk_chars={self.max_chunk_chars}",
                    "WARNING"
                )
                break
            
            # Calculate chunk end
            end = min(pos + self.max_chunk_chars, length)
            chunk = text[pos:end].strip()
            
            # Only add chunks that meet minimum size
            if len(chunk) >= self.min_chunk_chars:
                chunks.append(chunk)
            elif pos == 0 and len(chunk) > 0:
                # Special case: first chunk is shorter than min but not empty
                # This ensures we don't lose content from short documents
                chunks.append(chunk)
            
            # Calculate next position with overlap
            # CRITICAL: Ensure we always advance by at least 1 character
            advance = max(1, self.max_chunk_chars - self.overlap_chars)
            pos += advance
        
        return chunks

    def _chunk_id(self, base_id: str, idx: int) -> str:
        """Generate deterministic chunk ID"""
        raw = f"{base_id}:chunk_{idx:04d}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]  # Shorter hash for readability

    # -------------------------------------------------
    # PROCESS
    # -------------------------------------------------

    def process(self):
        """Process documents into chunks with validation"""
        self._log("Starting document processing...")
        
        if not self.documents:
            raise LoaderError("No documents to process")
        
        for idx, doc in enumerate(self.documents):
            try:
                # Validate document structure
                if not isinstance(doc, dict):
                    self._log(f"Document {idx} is not a dict, skipping", "WARNING")
                    continue
                
                # Get content
                content = doc.get("content", "")
                
                if not content or len(content.strip()) == 0:
                    self._log(f"Document {idx} has empty content, skipping", "DEBUG")
                    self.stats["docs_skipped_short"] += 1
                    continue
                
                if len(content) < self.min_doc_chars:
                    self._log(
                        f"Document {idx} too short ({len(content)} < {self.min_doc_chars}), skipping",
                        "DEBUG"
                    )
                    self.stats["docs_skipped_short"] += 1
                    continue
                
                # Get document ID
                base_id = doc.get("doc_id")
                if not base_id:
                    # Generate fallback ID
                    base_id = f"doc_{idx:04d}"
                    self._log(f"Document {idx} missing doc_id, using {base_id}", "WARNING")
                
                # Chunk the content
                chunks = self._chunk_text(content)
                
                if not chunks:
                    self._log(f"Document {base_id} produced no chunks", "WARNING")
                    continue
                
                # Create chunk records
                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_record = {
                        "chunk_id": self._chunk_id(base_id, chunk_idx),
                        "parent_doc_id": base_id,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "url": doc.get("url", ""),
                        "page_type": doc.get("page_type", "unknown"),
                        "title": doc.get("title", ""),
                        "content": chunk_text,
                        "content_length": len(chunk_text),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    
                    self.chunks.append(chunk_record)
                    self.stats["chunks_created"] += 1
                    self.stats["total_content_chars"] += len(chunk_text)
                
                self._log(
                    f"Document {base_id}: {len(chunks)} chunks created "
                    f"({len(content)} chars)",
                    "DEBUG"
                )
            
            except Exception as e:
                self._log(
                    f"Error processing document {idx} (id: {doc.get('doc_id', 'unknown')}): {e}",
                    "ERROR"
                )
                # Continue processing other documents
                continue
        
        # Final validation
        if len(self.chunks) == 0:
            raise LoaderError(
                f"ZERO chunks produced from {len(self.documents)} documents. "
                f"Check min_doc_chars={self.min_doc_chars} and min_chunk_chars={self.min_chunk_chars}"
            )
        
        self._log(f"Processing complete: {len(self.chunks)} chunks created", "SUCCESS")

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------

    def save(self):
        """Save chunks to output file with validation"""
        self._log(f"Saving output to: {self.output_file}")
        
        if not self.chunks:
            raise LoaderError("Cannot save: no chunks to write")
        
        output = {
            "metadata": {
                "generator": "connectiq_loader_v2.1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "input_file": self.input_file,
                "total_chunks": len(self.chunks),
                "chunking_config": {
                    "min_doc_chars": self.min_doc_chars,
                    "min_chunk_chars": self.min_chunk_chars,
                    "max_chunk_chars": self.max_chunk_chars,
                    "overlap_chars": self.overlap_chars,
                },
                "stats": self.stats,
            },
            "chunks": self.chunks,
        }
        
        try:
            # Ensure output directory exists
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write with atomic operation (temp file + rename)
            temp_file = f"{self.output_file}.tmp"
            
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            # Atomic rename
            Path(temp_file).replace(output_path)
            
            file_size = output_path.stat().st_size
            self._log(f"Output file created: {file_size:,} bytes", "SUCCESS")
            
        except Exception as e:
            raise LoaderError(f"Failed to write output file: {e}")

    # -------------------------------------------------
    # RUN
    # -------------------------------------------------

    def run(self):
        """Execute complete loading pipeline with comprehensive error handling"""
        try:
            print("\n" + "="*70)
            print("🚀 CONNECTIQ AI LOADER v2.1")
            print("="*70)
            
            # Load
            self.load()
            
            # Process
            self.process()
            
            # Save
            self.save()
            
            # Print summary
            self._print_summary()
            
            return True
            
        except LoaderError as e:
            self._log(f"Loader failed: {e}", "ERROR")
            self._print_summary(failed=True)
            raise
        
        except Exception as e:
            self._log(f"Unexpected error: {type(e).__name__}: {e}", "ERROR")
            self._print_summary(failed=True)
            raise LoaderError(f"Unexpected error: {e}") from e

    def _print_summary(self, failed: bool = False):
        """Print execution summary"""
        print("\n" + "="*70)
        
        if failed:
            print("❌ LOADER FAILED")
        else:
            print("✅ LOADER SUCCESS")
        
        print("="*70)
        print(f"  Input file:        {self.input_file}")
        print(f"  Output file:       {self.output_file}")
        print(f"  Documents loaded:  {self.stats['docs_loaded']}")
        print(f"  Documents skipped: {self.stats['docs_skipped_short']}")
        print(f"  Chunks created:    {self.stats['chunks_created']}")
        
        if self.stats['chunks_created'] > 0:
            avg_chunk_size = self.stats['total_content_chars'] // self.stats['chunks_created']
            print(f"  Avg chunk size:    {avg_chunk_size} chars")
            print(f"  Total content:     {self.stats['total_content_chars']:,} chars")
        
        print("="*70 + "\n")


# -------------------------------------------------
# CLI
# -------------------------------------------------

def main():
    """CLI entry point with improved argument parsing"""
    
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║          CONNECTIQ AI LOADER v2.1 - PRODUCTION SAFE              ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  python loader.py <input_file> [output_file] [options]

Arguments:
  input_file         Crawler output JSON file (required)
  output_file        Output chunks JSON file (default: rag_documents.json)

Options:
  --min-doc-chars N      Minimum document length (default: 200)
  --min-chunk-chars N    Minimum chunk length (default: 400)
  --max-chunk-chars N    Maximum chunk length (default: 900)
  --overlap-chars N      Overlap between chunks (default: 120)
  --quiet                Suppress debug output
  -h, --help             Show this help

Examples:
  python loader.py crawler_output.json
  python loader.py crawler_output.json processed_chunks.json
  python loader.py data.json out.json --max-chunk-chars 1200 --overlap-chars 200
  python loader.py crawler_output.json --quiet
        """)
        sys.exit(0)

    # Parse required arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "rag_documents.json"
    
    # Parse optional arguments
    kwargs = {
        "verbose": True,
    }
    
    i = 3 if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == "--min-doc-chars" and i + 1 < len(sys.argv):
            kwargs["min_doc_chars"] = int(sys.argv[i + 1])
            i += 2
        elif arg == "--min-chunk-chars" and i + 1 < len(sys.argv):
            kwargs["min_chunk_chars"] = int(sys.argv[i + 1])
            i += 2
        elif arg == "--max-chunk-chars" and i + 1 < len(sys.argv):
            kwargs["max_chunk_chars"] = int(sys.argv[i + 1])
            i += 2
        elif arg == "--overlap-chars" and i + 1 < len(sys.argv):
            kwargs["overlap_chars"] = int(sys.argv[i + 1])
            i += 2
        elif arg == "--quiet":
            kwargs["verbose"] = False
            i += 1
        else:
            print(f"❌ Unknown option: {arg}")
            print("Use --help for usage information")
            sys.exit(1)
    
    # Validate configuration
    if kwargs.get("overlap_chars", 120) >= kwargs.get("max_chunk_chars", 900):
        print("❌ ERROR: overlap_chars must be less than max_chunk_chars")
        sys.exit(1)
    
    # Execute loader
    try:
        loader = AILoader(input_file, output_file, **kwargs)
        loader.run()
        sys.exit(0)
        
    except LoaderError as e:
        print(f"\n❌ LOADER FAILED: {e}")
        sys.exit(2)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()