# run_pipeline.py
"""
ConnectIQ RAG Pipeline Runner
==============================
Orchestrates: Crawler → Loader → Embeddings

Usage:
    python run_pipeline.py <url> [--max-pages N]

Example:
    python run_pipeline.py https://nfweek.com --max-pages 60
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from crawler import ProductionCrawler
from loader import AILoader


def main():
    parser = argparse.ArgumentParser(
        description="Run ConnectIQ RAG data pipeline"
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
    
    print("\n" + "="*70)
    print("🚀 CONNECTIQ RAG PIPELINE")
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
    print("📡 STEP 1/2: Crawling website...")
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
    print("📦 STEP 2/2: Loading and chunking...")
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
    # SUMMARY
    # ==========================================
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("="*70)
    print("📊 PIPELINE SUMMARY")
    print("="*70)
    print(f"✅ Documents crawled: {len(documents)}")
    print(f"✅ Chunks created: {len(loader.chunks)}")
    print(f"✅ Time elapsed: {elapsed:.1f}s")
    print(f"\n📁 Output files:")
    print(f"   - {crawler_output}")
    print(f"   - {rag_output}")
    print("\n💡 Next step: Generate embeddings")
    print(f"   python embeddings.py --input {rag_output}")
    print("="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    main()