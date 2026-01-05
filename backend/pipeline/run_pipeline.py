# run_pipeline.py
"""
ConnectIQ RAG Pipeline Runner - ATOMIC VERSION
===============================================
Orchestrates: Crawler → Loader → Embeddings (ALL IN ONE COMMAND)

Usage:
    python run_pipeline.py <url> [--max-pages N]

Example:
    python run_pipeline.py https://nfweek.com --max-pages 60

Output:
    ✅ PIPELINE READY - System ready for chat queries
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from crawler import ProductionCrawler
from loader import AILoader
from embeddings import EmbeddingPipeline  # ← NEW IMPORT


def main():
    parser = argparse.ArgumentParser(
        description="Run ConnectIQ RAG data pipeline (ATOMIC)"
    )
    parser.add_argument(
        "url",
        help="Website URL to crawl"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=60,
        help="Maximum pages to crawl (default: 60)"
    )
    parser.add_argument(
        "--event-id",
        default="nfweek2026",
        help="Event ID for chunks (default: nfweek2026)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)
    
    crawler_output = DATA_DIR / "crawler_output.json"
    rag_output = DATA_DIR / "rag_documents.json"
    embeddings_output = DATA_DIR / "embeddings.json"  # ← NEW
    
    print("\n" + "="*70)
    print("🚀 CONNECTIQ ATOMIC RAG PIPELINE")
    print("="*70)
    print(f"Target URL: {args.url}")
    print(f"Max pages: {args.max_pages}")
    print(f"Event ID: {args.event_id}")
    print(f"Output dir: {DATA_DIR.absolute()}")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    
    # ==========================================
    # STEP 1: CRAWL
    # ==========================================
    print("📡 STEP 1/3: Crawling website...")
    print("-" * 70)
    
    try:
        crawler = ProductionCrawler(
            start_url=args.url,
            max_pages=args.max_pages
        )
        documents = crawler.crawl()
        crawler.save(crawler_output)
        
        if not documents:
            print("❌ Error: No documents crawled")
            sys.exit(1)
        
        print(f"✅ Crawl completed: {len(documents)} documents\n")
        
    except Exception as e:
        print(f"❌ Crawl failed: {e}")
        sys.exit(1)
    
    # ==========================================
    # STEP 2: LOAD & CHUNK
    # ==========================================
    print("📦 STEP 2/3: Loading and chunking...")
    print("-" * 70)
    
    try:
        loader = AILoader(
            input_file=str(crawler_output),
            output_file=str(rag_output),
            event_id=args.event_id
        )
        loader.run()
        
        if not loader.chunks:
            print("❌ Error: No chunks created")
            sys.exit(1)
        
        print(f"✅ Loading completed: {len(loader.chunks)} chunks\n")
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        sys.exit(1)
    
    # ==========================================
    # STEP 3: GENERATE EMBEDDINGS (NEW!)
    # ==========================================
    print("🧬 STEP 3/3: Generating embeddings...")
    print("-" * 70)
    
    try:
        embedding_pipeline = EmbeddingPipeline(
            input_file=str(rag_output),
            output_file=str(embeddings_output),
            verbose=True  # Show progress
        )
        embedding_pipeline.run()
        
        if not embedding_pipeline.embeddings:
            print("❌ Error: No embeddings generated")
            sys.exit(1)
        
        print(f"✅ Embeddings completed: {len(embedding_pipeline.embeddings)} vectors\n")
        
    except Exception as e:
        print(f"❌ Embeddings failed: {e}")
        print(f"💡 Make sure OPENAI_API_KEY is set in .env file")
        sys.exit(1)
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("="*70)
    print("✅ PIPELINE READY")
    print("="*70)
    print(f"📊 Pipeline Stats:")
    print(f"   Documents crawled: {len(documents)}")
    print(f"   Chunks created: {len(loader.chunks)}")
    print(f"   Embeddings generated: {len(embedding_pipeline.embeddings)}")
    print(f"   Time elapsed: {elapsed:.1f}s")
    print(f"\n📁 Output files:")
    print(f"   ✅ {crawler_output}")
    print(f"   ✅ {rag_output}")
    print(f"   ✅ {embeddings_output}")
    print(f"\n🎉 System is READY for chat queries!")
    print(f"   Start backend: cd backend && uvicorn main:app --reload")
    print("="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    main()