# loader.py
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

class AILoader:
    """
    Smart document loader with intelligent chunking strategy.
    
    Key improvements:
    - Keeps short pages as single chunks (preserves context)
    - Smart overlap for long documents
    - Preserves metadata and page types
    """
    
    def __init__(
        self, 
        input_file: str, 
        output_file: str, 
        event_id: str | None = None,
        short_page_threshold: int = 1500,  # NEW
        chunk_size: int = 900,             # NEW
        chunk_overlap: int = 100,          # NEW
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.event_id = event_id
        
        # Chunking config
        self.short_page_threshold = short_page_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.chunks = []
        self.stats = {
            "total_docs": 0,
            "single_chunk_docs": 0,
            "multi_chunk_docs": 0,
            "total_chunks": 0,
        }

    def load(self):
        """Load documents from crawler output"""
        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Support both formats
        if "documents" in data:
            self.docs = data["documents"]
        else:
            self.docs = data
        
        self.stats["total_docs"] = len(self.docs)
        print(f"📂 Loaded {len(self.docs)} documents")

    def _create_chunks_with_overlap(self, content: str) -> List[str]:
        """
        Create overlapping chunks for long content.
        
        Strategy:
        - Chunk size: 900 chars
        - Overlap: 100 chars (preserves context between chunks)
        """
        if len(content) <= self.chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.chunk_size
            chunk = content[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
            
            # Move forward with overlap
            start += (self.chunk_size - self.chunk_overlap)
        
        return chunks

    def process(self):
        """
        Process documents with intelligent chunking strategy.
        
        CRITICAL LOGIC:
        - Short pages (< 1500 chars) → 1 chunk (preserves semantic unity)
        - Long pages → multiple chunks with overlap
        """
        for doc in self.docs:
            content = doc["content"]
            content_length = len(content)
            
            # STRATEGY 1: Short pages stay as single chunk
            if content_length < self.short_page_threshold:
                chunks = [content]
                self.stats["single_chunk_docs"] += 1
            
            # STRATEGY 2: Long pages get chunked with overlap
            else:
                chunks = self._create_chunks_with_overlap(content)
                self.stats["multi_chunk_docs"] += 1
            
            # Create chunk objects
            for i, chunk_content in enumerate(chunks):
                chunk_id = hashlib.sha256(
                    f"{doc['doc_id']}_{i}".encode()
                ).hexdigest()[:16]
                
                self.chunks.append({
                    "chunk_id": chunk_id,
                    "parent_doc_id": doc["doc_id"],
                    "content": chunk_content,
                    "page_type": doc.get("page_type", "general"),
                    "event_id": self.event_id,
                    "url": doc["url"],
                    "title": doc.get("title", ""),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "is_single_chunk": len(chunks) == 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
            
            self.stats["total_chunks"] += len(chunks)
        
        print(f"📊 Chunking stats:")
        print(f"   Single-chunk docs: {self.stats['single_chunk_docs']}")
        print(f"   Multi-chunk docs: {self.stats['multi_chunk_docs']}")
        print(f"   Total chunks: {self.stats['total_chunks']}")
        print(f"   Avg chunks per doc: {self.stats['total_chunks'] / self.stats['total_docs']:.1f}")

    def save(self):
        """Save chunks with metadata"""
        Path(self.output_file).parent.mkdir(exist_ok=True)
        
        output = {
            "metadata": {
                "event_id": self.event_id,
                "total_chunks": len(self.chunks),
                "stats": self.stats,
                "chunking_config": {
                    "short_page_threshold": self.short_page_threshold,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "chunks": self.chunks
        }
        
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(self.chunks)} chunks to {self.output_file}")

    def run(self):
        """Execute full loading pipeline"""
        self.load()
        self.process()
        self.save()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python loader.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/rag_documents.json"
    
    loader = AILoader(
        input_file=input_file,
        output_file=output_file,
        event_id="nfweek2026"
    )
    loader.run()