"""
CONNECTIQ PRODUCTION CRAWLER v2.0
===================================
Enterprise-grade web crawler optimized for AI/RAG ingestion.
Built for reliability, deduplication, and semantic extraction.

Author: ConnectIQ Team
License: Proprietary - Commercial Use
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.robotparser import RobotFileParser
from collections import deque
from datetime import datetime, timezone
from difflib import SequenceMatcher
import re
import json
import sys
import hashlib
import time
from typing import List, Dict, Optional, Set


class ProductionCrawler:
    """
    AI-first web crawler with enterprise features:
    - Rate limiting & retry logic
    - Robots.txt compliance
    - Advanced deduplication (hash + similarity)
    - URL prioritization (BFS with scoring)
    - Robust error handling
    - Semantic content extraction
    """

    def __init__(
        self,
        start_url: str,
        max_pages: int = 40,
        timeout: int = 12,
        min_content_length: int = 200,  # Lowered from 250 to catch speaker pages
        rate_limit_delay: float = 0.5,
        similarity_threshold: float = 0.85,
        respect_robots: bool = True,
        user_agent: str = "ConnectIQ-Crawler/2.0 (+https://connectiq.ai/bot)",
    ):
        self.start_url = start_url.rstrip("/")
        parsed = urlparse(self.start_url)

        self.domain = parsed.netloc
        self.scheme = parsed.scheme
        self.base_url = f"{self.scheme}://{self.domain}"

        # Config
        self.max_pages = max_pages
        self.timeout = timeout
        self.min_content_length = min_content_length
        self.rate_limit_delay = rate_limit_delay
        self.similarity_threshold = similarity_threshold
        self.respect_robots = respect_robots
        self.user_agent = user_agent

        # State
        self.visited: Set[str] = set()
        self.queued: Set[str] = set([self.start_url])
        self.queue: deque = deque([(self.start_url, 0)])  # (url, priority_score)
        self.documents: List[Dict] = []
        self.content_hashes: Set[str] = set()
        self.content_signatures: List[str] = []  # For similarity check

        # Stats
        self.stats = {
            "visited": 0,
            "stored": 0,
            "skipped_short": 0,
            "skipped_duplicate": 0,
            "skipped_similar": 0,
            "skipped_non_html": 0,
            "skipped_robots": 0,
            "errors": 0,
        }

        # Setup
        self._setup_session()
        self._compile_filters()
        self._load_robots_txt()

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def _setup_session(self):
        """Configure requests session with retry logic and connection pooling"""
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": self.user_agent})

    def _compile_filters(self):
        """Compile regex patterns for URL filtering"""
        self.excluded_path_patterns = re.compile(
            r"(privacy|cookie|terms|conditions|login|account|signup|sign-up|register|"
            r"press|news|blog|career|jobs|sponsor|download|print|share|rss|feed)",
            re.IGNORECASE,
        )

        self.static_path_patterns = re.compile(
            r"(/wp-content/|/wp-includes/|/wp-json/|/assets/|/static/|/media/|"
            r"/uploads/|/files/|/downloads/|/css/|/js/|/images/|/img/)",
            re.IGNORECASE,
        )

        self.excluded_extensions = (
            ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp",
            ".css", ".js", ".woff", ".woff2", ".ttf", ".eot", ".otf",
            ".mp4", ".avi", ".mov", ".mp3", ".wav", ".ogg", ".webm",
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".zip", ".rar", ".tar", ".gz", ".7z",
            ".xml", ".json", ".txt", ".csv",
        )

        # Query params to strip (tracking, sessions)
        self.strip_params = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "gclid", "ref", "source", "sessionid", "sid",
        }

    def _load_robots_txt(self):
        """Load and parse robots.txt"""
        self.robots_parser = None
        
        if not self.respect_robots:
            return
        
        try:
            robots_url = f"{self.base_url}/robots.txt"
            response = self.session.get(robots_url, timeout=5)
            
            if response.status_code == 200:
                self.robots_parser = RobotFileParser()
                self.robots_parser.parse(response.text.splitlines())
                print(f"✓ Loaded robots.txt from {robots_url}")
        except:
            print(f"⚠ Could not load robots.txt, proceeding without restrictions")

    # ==================================================================
    # URL PROCESSING
    # ==================================================================

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing tracking params and fragments"""
        parsed = urlparse(url)
        
        # Strip query params
        if parsed.query:
            params = parse_qs(parsed.query)
            filtered_params = {
                k: v for k, v in params.items() 
                if k.lower() not in self.strip_params
            }
            # Rebuild query string
            query = "&".join(f"{k}={v[0]}" for k, v in filtered_params.items())
        else:
            query = ""
        
        # Rebuild URL without fragment
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if query:
            normalized += f"?{query}"
        
        return normalized.rstrip("/")

    def _calculate_url_priority(self, url: str) -> int:
        """Calculate URL crawl priority (higher = more important)"""
        path = urlparse(url).path.lower()
        score = 0
        
        # High priority keywords
        if any(kw in path for kw in ["speaker", "agenda", "program", "schedule"]):
            score += 10
        
        # Medium priority
        if any(kw in path for kw in ["about", "venue", "location", "faq", "ticket"]):
            score += 5
        
        # Penalize deep paths
        depth = path.count("/")
        score -= depth
        
        return score

    def _is_valid_url(self, url: str) -> bool:
        """Comprehensive URL validation"""
        parsed = urlparse(url)

        # Basic checks
        if parsed.scheme not in ("http", "https"):
            return False
        
        if parsed.netloc != self.domain:
            return False
        
        if url in self.visited or url in self.queued:
            return False

        # Pattern exclusions
        if self.excluded_path_patterns.search(parsed.path):
            return False
        
        if self.static_path_patterns.search(parsed.path):
            return False
        
        if parsed.path.lower().endswith(self.excluded_extensions):
            return False

        # Robots.txt check
        if self.robots_parser and not self.robots_parser.can_fetch(self.user_agent, url):
            self.stats["skipped_robots"] += 1
            return False

        return True

    # ==================================================================
    # CONTENT FETCHING
    # ==================================================================

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML with rate limiting and error handling"""
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            
            if response.status_code != 200:
                return None

            # Verify content type
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                self.stats["skipped_non_html"] += 1
                return None

            # Handle encoding
            if response.encoding is None:
                response.encoding = response.apparent_encoding

            return response.text

        except requests.RequestException as e:
            self.stats["errors"] += 1
            print(f"  ⚠ Fetch error: {type(e).__name__}")
            return None

    # ==================================================================
    # CONTENT EXTRACTION
    # ==================================================================

    def _extract_links(self, html: str, base_url: str) -> List[tuple]:
        """Extract and prioritize links"""
        soup = BeautifulSoup(html, "html.parser")
        links = []

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            full_url = urljoin(base_url, href)
            full_url = self._normalize_url(full_url)

            if full_url not in self.queued and self._is_valid_url(full_url):
                priority = self._calculate_url_priority(full_url)
                self.queued.add(full_url)
                links.append((full_url, priority))

        return links

    def _extract_content(self, html: str) -> Dict:
        """Extract semantic content from HTML"""
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "iframe"]):
            tag.decompose()

        # Title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Meta description
        meta_desc = ""
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            meta_desc = meta["content"].strip()

        # Headings
        headings = []
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
            text = h.get_text(strip=True)
            if text and len(text) > 3:
                headings.append(text)

        # Body content
        body_blocks = []
        
        # Main content area (prioritize semantic HTML5)
        main_content = soup.find("main") or soup.find("article") or soup
        
        for p in main_content.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 40:
                body_blocks.append(text)

        for li in main_content.find_all("li"):
            text = li.get_text(strip=True)
            if 15 < len(text) < 400:
                body_blocks.append(f"• {text}")

        # Combine content
        content_parts = []
        if meta_desc:
            content_parts.append(f"[META] {meta_desc}\n")
        if headings:
            content_parts.append("\n".join(headings) + "\n")
        content_parts.extend(body_blocks)
        
        content = "\n".join(content_parts)
        content = re.sub(r"\s+", " ", content)  # Normalize whitespace
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        return {
            "title": title,
            "meta_description": meta_desc,
            "content": content,
            "content_length": len(content),
            "headings_count": len(headings),
        }

    # ==================================================================
    # DEDUPLICATION
    # ==================================================================

    def _hash_content(self, content: str) -> str:
        """Generate SHA256 hash for exact duplicate detection"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _is_similar_content(self, content: str) -> bool:
        """Check if content is similar to existing documents (fuzzy deduplication)"""
        if not self.content_signatures:
            return False
        
        # Create signature (first 1000 chars for better accuracy)
        signature = content[:1000]
        
        for existing_sig in self.content_signatures[-30:]:  # Check last 30 docs
            similarity = SequenceMatcher(None, signature, existing_sig).ratio()
            if similarity > self.similarity_threshold:
                return True
        
        return False

    # ==================================================================
    # PAGE CLASSIFICATION
    # ==================================================================

    def _infer_page_type(self, url: str, title: str, content: str) -> str:
        """Infer page type from URL, title and content"""
        text = (url + " " + title + " " + content[:200]).lower()
        
        if "speaker" in text:
            return "speakers"
        if any(kw in text for kw in ["agenda", "program", "schedule"]):
            return "program"
        if any(kw in text for kw in ["ticket", "price", "registration", "register"]):
            return "tickets"
        if "faq" in text or "question" in text:
            return "faq"
        if any(kw in text for kw in ["venue", "location", "hotel", "travel"]):
            return "location"
        if "about" in text:
            return "about"
        if "sponsor" in text or "partner" in text:
            return "sponsors"
        
        return "general"

    # ==================================================================
    # MAIN CRAWL LOOP
    # ==================================================================

    def crawl(self) -> List[Dict]:
        """Execute crawl with prioritized BFS"""
        print("\n" + "="*70)
        print("🚀 CONNECTIQ PRODUCTION CRAWLER v2.0")
        print("="*70)
        print(f"Target: {self.start_url}")
        print(f"Max pages: {self.max_pages}")
        print(f"Rate limit: {self.rate_limit_delay}s")
        print(f"Robots.txt: {'✓ Enabled' if self.respect_robots else '✗ Disabled'}")
        print("="*70 + "\n")

        start_time = time.time()

        while self.queue and len(self.documents) < self.max_pages:
            # Get highest priority URL
            self.queue = deque(sorted(self.queue, key=lambda x: -x[1]))
            url, priority = self.queue.popleft()
            
            self.visited.add(url)
            self.stats["visited"] += 1

            print(f"[{len(self.documents)+1}/{self.max_pages}] (P:{priority:+d}) {url}")

            # Fetch
            html = self._fetch_html(url)
            if not html:
                continue

            # Extract
            content_data = self._extract_content(html)
            
            # Validate length
            if content_data["content_length"] < self.min_content_length:
                self.stats["skipped_short"] += 1
                print(f"  ⚠ Too short ({content_data['content_length']} chars)")
                continue

            # Check exact duplicates
            content_hash = self._hash_content(content_data["content"])
            if content_hash in self.content_hashes:
                self.stats["skipped_duplicate"] += 1
                print(f"  ⚠ Duplicate (hash match)")
                continue

            # Check similar content
            if self._is_similar_content(content_data["content"]):
                self.stats["skipped_similar"] += 1
                print(f"  ⚠ Similar content (>85% match)")
                continue

            # Store document
            document = {
                "doc_id": f"doc_{len(self.documents):04d}",
                "url": url,
                "page_type": self._infer_page_type(url, content_data["title"], content_data["content"]),
                "title": content_data["title"],
                "meta_description": content_data["meta_description"],
                "content": content_data["content"],
                "content_hash": content_hash,
                "content_length": content_data["content_length"],
                "headings_count": content_data["headings_count"],
                "priority_score": priority,
                "source": "crawler_v2",
                "crawled_at": datetime.now(timezone.utc).isoformat(),
            }

            self.documents.append(document)
            self.content_hashes.add(content_hash)
            self.content_signatures.append(content_data["content"][:1000])
            self.stats["stored"] += 1
            
            print(f"  ✅ Stored ({content_data['content_length']} chars, {content_data['headings_count']} headings, type: {document['page_type']})")

            # Queue new links
            for link, link_priority in self._extract_links(html, url):
                self.queue.append((link, link_priority))

        elapsed = time.time() - start_time
        self._print_summary(elapsed)
        
        return self.documents

    # ==================================================================
    # OUTPUT
    # ==================================================================

    def save(self, filepath: str):
        """Save crawled documents with metadata"""
        output = {
            "metadata": {
                "crawler_version": "2.0",
                "start_url": self.start_url,
                "domain": self.domain,
                "crawled_at": datetime.now(timezone.utc).isoformat(),
                "total_documents": len(self.documents),
                "stats": self.stats,
            },
            "documents": self.documents,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        total_chars = sum(d["content_length"] for d in self.documents)
        avg_chars = total_chars // len(self.documents) if self.documents else 0

        print(f"\n💾 OUTPUT SAVED")
        print(f"   File: {filepath}")
        print(f"   Documents: {len(self.documents)}")
        print(f"   Total content: {total_chars:,} chars")
        print(f"   Average per doc: {avg_chars:,} chars")

    def _print_summary(self, elapsed: float):
        """Print crawl statistics"""
        print("\n" + "="*70)
        print("📊 CRAWL SUMMARY")
        print("="*70)
        
        for key, value in self.stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        efficiency = (self.stats["stored"] / self.stats["visited"] * 100) if self.stats["visited"] > 0 else 0
        
        print(f"\n  Efficiency: {efficiency:.1f}% (stored/visited)")
        print(f"  Elapsed time: {elapsed:.1f}s")
        print(f"  Avg time per page: {elapsed/self.stats['visited']:.2f}s" if self.stats['visited'] > 0 else "")
        print("="*70)


# ======================================================================
# CLI INTERFACE
# ======================================================================

def main():
    if len(sys.argv) < 2:
        print("""
╔═══════════════════════════════════════════════════════════════╗
║          CONNECTIQ PRODUCTION CRAWLER v2.0                    ║
╚═══════════════════════════════════════════════════════════════╝

Usage:
  python site_crawler.py <url> [options]

Options:
  --max-pages N        Maximum pages to crawl (default: 40)
  --rate-limit N       Delay between requests in seconds (default: 0.5)
  --no-robots          Ignore robots.txt
  --output FILE        Output filename (default: crawler_output.json)

Examples:
  python site_crawler.py https://nfweek.com
  python site_crawler.py https://conference.ai --max-pages 100 --rate-limit 1.0
  python site_crawler.py https://event.com --no-robots --output data.json
        """)
        sys.exit(1)

    # Parse arguments
    url = sys.argv[1]
    max_pages = 40
    rate_limit = 0.5
    respect_robots = True
    output_file = "crawler_output.json"

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--max-pages" and i + 1 < len(sys.argv):
            max_pages = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--rate-limit" and i + 1 < len(sys.argv):
            rate_limit = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--no-robots":
            respect_robots = False
            i += 1
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        else:
            print(f"Unknown option: {sys.argv[i]}")
            sys.exit(1)

    # Execute crawl
    crawler = ProductionCrawler(
        start_url=url,
        max_pages=max_pages,
        rate_limit_delay=rate_limit,
        respect_robots=respect_robots,
    )

    documents = crawler.crawl()
    crawler.save(output_file)


if __name__ == "__main__":
    main()