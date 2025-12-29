# crawler.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path

class ProductionCrawler:
    def __init__(
        self,
        start_url: str,
        max_pages: int = 60,
        min_content_length: int = 100,
    ):
        self.start_url = start_url.rstrip("/")
        self.domain = urlparse(self.start_url).netloc
        self.max_pages = max_pages
        self.min_content_length = min_content_length

        self.visited = set()
        self.queue = deque([self.start_url])
        self.documents = []
        
        # NEW: Semantic enrichment templates
        self.semantic_enrichment = {
            "partners": "This page lists official partners, sponsors, and supporting organizations of the conference. ",
            "sponsors": "This page contains information about event sponsors, their sponsorship levels, and contributions. ",
            "about": "This page describes the conference organization, mission, values, and background information. ",
            "team": "This page lists staff members, organizers, project managers, and key organizational contacts. ",
            "speakers": "This page contains information about conference speakers, their backgrounds, and talk topics. ",
        }

    def _infer_page_type(self, url: str, content_preview: str = "") -> str:
        """
        Infer page type from URL and content.
        
        IMPROVED: Also checks content for better classification
        """
        path = urlparse(url).path.lower()
        preview = content_preview.lower()
        
        # Check URL path first
        if "partner" in path or "sponsor" in path:
            return "partners"
        if "about" in path or "team" in path or "staff" in path:
            return "about"
        if "speaker" in path:
            return "speakers"
        if "ticket" in path or "faq" in path:
            return "tickets"
        if "agenda" in path or "program" in path or "schedule" in path:
            return "program"
        
        # Check content if path is ambiguous
        if any(kw in preview for kw in ["project manager", "organizer", "team member", "contact"]):
            return "team"
        if any(kw in preview for kw in ["sponsor", "partner", "supporter"]):
            return "sponsors"
        
        return "general"
    
    def _enrich_content(self, content: str, page_type: str) -> str:
        """
        Add semantic context to specific page types.
        
        CRITICAL: This dramatically improves retrieval for staff/partner queries
        """
        # Apply semantic enrichment if page type matches
        if page_type in self.semantic_enrichment:
            prefix = self.semantic_enrichment[page_type]
            return prefix + content
        
        return content

    def crawl(self):
        while self.queue and len(self.documents) < self.max_pages:
            url = self.queue.popleft()
            if url in self.visited:
                continue
            self.visited.add(url)

            try:
                r = requests.get(url, timeout=10)
                if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                    continue
            except:
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            
            # Remove noise
            for t in soup(["script", "style", "noscript"]):
                t.decompose()

            # Extract raw content
            raw_content = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
            
            # Skip if too short
            if len(raw_content) < self.min_content_length:
                continue

            # Infer page type (with content preview for better accuracy)
            page_type = self._infer_page_type(url, raw_content[:500])
            
            # Apply semantic enrichment
            enriched_content = self._enrich_content(raw_content, page_type)
            
            # Generate hash of ORIGINAL content (not enriched)
            h = hashlib.sha256(raw_content.encode()).hexdigest()

            self.documents.append({
                "doc_id": f"doc_{len(self.documents):04d}",
                "url": url,
                "title": soup.title.string.strip() if soup.title and soup.title.string else "",
                "content": enriched_content,  # Store enriched version
                "page_type": page_type,
                "content_hash": h,
                "content_length": len(enriched_content),
                "original_length": len(raw_content),
                "enriched": page_type in self.semantic_enrichment,
                "crawled_at": datetime.now(timezone.utc).isoformat(),
            })

            # Queue new links
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                if urlparse(link).netloc == self.domain:
                    self.queue.append(link)

        return self.documents

    def save(self, path: Path):
        """Save with metadata"""
        path.parent.mkdir(exist_ok=True)
        
        output = {
            "metadata": {
                "start_url": self.start_url,
                "total_documents": len(self.documents),
                "enriched_documents": sum(1 for d in self.documents if d.get("enriched")),
                "crawled_at": datetime.now(timezone.utc).isoformat(),
            },
            "documents": self.documents
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(self.documents)} documents")
        print(f"   Enriched: {output['metadata']['enriched_documents']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python crawler.py <url>")
        sys.exit(1)
    
    out = Path("data/crawler_output.json")
    crawler = ProductionCrawler(sys.argv[1])
    crawler.crawl()
    crawler.save(out)