import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from collections import deque


class SiteCrawler:
    def __init__(
        self,
        start_url: str,
        max_pages: int = 30,
        timeout: int = 10,
    ):
        self.start_url = start_url.rstrip("/")
        self.domain = urlparse(self.start_url).netloc
        self.scheme = urlparse(self.start_url).scheme
        self.max_pages = max_pages
        self.timeout = timeout

        self.visited = set()
        self.to_visit = deque([self.start_url])
        self.relevant_pages = []

        # Hard exclusions (automatic guardrails)
        self.excluded_patterns = re.compile(
            r"(privacy|cookie|terms|login|account|signup|sign-up|press|blog|career|jobs|sponsor)",
            re.IGNORECASE,
        )

    def is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc != self.domain:
            return False
        if self.excluded_patterns.search(parsed.path):
            return False
        if url in self.visited:
            return False
        return True

    def fetch_page(self, url: str) -> str | None:
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "ConnectIQ-Crawler/1.0"},
            )
            if response.status_code == 200:
                return response.text
        except requests.RequestException:
            pass
        return None

    def extract_links(self, html: str, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links = []

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("#"):
                continue
            full_url = urljoin(base_url, href)
            full_url = full_url.split("#")[0].rstrip("/")
            links.append(full_url)

        return links

    def crawl(self) -> list[str]:
        while self.to_visit and len(self.relevant_pages) < self.max_pages:
            current_url = self.to_visit.popleft()

            if not self.is_valid_url(current_url):
                continue

            html = self.fetch_page(current_url)
            self.visited.add(current_url)

            if not html:
                continue

            self.relevant_pages.append(current_url)

            for link in self.extract_links(html, current_url):
                if self.is_valid_url(link):
                    self.to_visit.append(link)

        return self.relevant_pages


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python site_crawler.py <conference_url>")
        sys.exit(1)

    start_url = sys.argv[1]
    crawler = SiteCrawler(start_url)
    pages = crawler.crawl()

    print("\nCrawled pages:")
    for p in pages:
        print("-", p)
