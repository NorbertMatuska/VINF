import requests
import time
import random
import csv
import os
import hashlib
from urllib.parse import urljoin, urlparse
from collections import deque
import re
import urllib.robotparser
from datetime import datetime
import json
from typing import List, Tuple, Optional, Dict, Any
import signal
import sys


class CSVCrawlerManager:
    def __init__(self, data_dir='csv_data'):
        self.data_dir = data_dir
        self.urls_file = os.path.join(data_dir, 'urls.csv')
        self.stats_file = os.path.join(data_dir, 'crawl_stats.csv')
        self.checkpoints_file = os.path.join(data_dir, 'crawl_checkpoints.csv')

        # In-memory cache for performance
        self._urls_cache = None
        self._urls_cache_modified = False
        self._stats_cache = None
        self._checkpoints_cache = None

        # Ensure directories and files exist
        self._initialize_files()

    def _initialize_files(self):
        """Ensure all CSV files exist with proper headers"""
        os.makedirs(self.data_dir, exist_ok=True)

        files_schemas = {
            self.urls_file: [
                'url', 'normalized_url', 'status', 'depth', 'discovered_at',
                'crawled_at', 'retry_count', 'file_path', 'response_code', 'error_message'
            ],
            self.stats_file: [
                'run_id', 'start_time', 'end_time', 'pages_crawled',
                'urls_discovered', 'total_size_mb'
            ],
            self.checkpoints_file: [
                'id', 'checkpoint_time', 'urls_crawled', 'urls_queued', 'total_urls'
            ]
        }

        for file_path, headers in files_schemas.items():
            if not os.path.exists(file_path):
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()

    def _load_urls_cache(self):
        """Load URLs into memory cache"""
        if self._urls_cache is None:
            self._urls_cache = {}
            try:
                with open(self.urls_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self._urls_cache[row['normalized_url']] = row
            except FileNotFoundError:
                pass

    def _save_urls_cache(self):
        """Save URLs cache to CSV"""
        if self._urls_cache_modified and self._urls_cache:
            with open(self.urls_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'url', 'normalized_url', 'status', 'depth', 'discovered_at',
                    'crawled_at', 'retry_count', 'file_path', 'response_code', 'error_message'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self._urls_cache.values():
                    writer.writerow(row)
            self._urls_cache_modified = False

    def url_exists(self, url: str) -> bool:
        """Check if URL exists using normalized URL"""
        self._load_urls_cache()
        normalized_url = self._normalize_url(url)
        return normalized_url in self._urls_cache

    def _normalize_url(self, url: str) -> str:
        """Normalize URL (copied from crawler logic)"""
        from urllib.parse import urlparse
        import re

        parsed = urlparse(url)
        scheme = 'https'
        netloc = parsed.netloc.lower()

        if netloc.startswith('www.'):
            netloc = netloc[4:]

        path = parsed.path.rstrip('/')
        if not path:
            path = '/'

        # Remove common tracking parameters
        query_params = []
        if parsed.query:
            for param in parsed.query.split('&'):
                if not any(track_param in param.lower() for track_param in
                           ['utm_', 'ref=', 'source=', 'campaign=', 'medium=']):
                    query_params.append(param)

        query = '&'.join(query_params) if query_params else ''

        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"

        return normalized

    def add_urls(self, urls: List[str], depth: int = 0) -> int:
        """Add new URLs to database with batch processing"""
        self._load_urls_cache()
        new_count = 0

        for url in urls:
            normalized_url = self._normalize_url(url)

            if normalized_url not in self._urls_cache:
                self._urls_cache[normalized_url] = {
                    'url': url,
                    'normalized_url': normalized_url,
                    'status': 'discovered',
                    'depth': str(depth),
                    'discovered_at': datetime.now().isoformat(),
                    'crawled_at': '',
                    'retry_count': '0',
                    'file_path': '',
                    'response_code': '',
                    'error_message': ''
                }
                new_count += 1

        if new_count > 0:
            self._urls_cache_modified = True
            self._save_urls_cache()

        return new_count

    def get_next_urls(self, limit: int = 50) -> List[tuple]:
        """Get next batch of URLs to crawl"""
        self._load_urls_cache()

        discovered_urls = []
        for row in self._urls_cache.values():
            if row['status'] == 'discovered':
                discovered_urls.append((row['url'], int(row['depth'])))

        # Sort by depth and discovery time
        discovered_urls.sort(key=lambda x: (x[1], x[0]))

        # Mark them as queued
        for url, depth in discovered_urls[:limit]:
            normalized_url = self._normalize_url(url)
            if normalized_url in self._urls_cache:
                self._urls_cache[normalized_url]['status'] = 'queued'

        self._urls_cache_modified = True
        self._save_urls_cache()

        return discovered_urls[:limit]

    def mark_url_crawled(self, url: str, file_path: str, response_code: int = 200):
        """Mark URL as successfully crawled"""
        self._load_urls_cache()
        normalized_url = self._normalize_url(url)

        if normalized_url in self._urls_cache:
            self._urls_cache[normalized_url].update({
                'status': 'crawled',
                'crawled_at': datetime.now().isoformat(),
                'file_path': file_path,
                'response_code': str(response_code)
            })
            self._urls_cache_modified = True
            self._save_urls_cache()

    def mark_url_failed(self, url: str, error_message: str = "", response_code: int = None):
        """Mark URL as failed after retries"""
        self._load_urls_cache()
        normalized_url = self._normalize_url(url)

        if normalized_url in self._urls_cache:
            self._urls_cache[normalized_url].update({
                'status': 'failed',
                'crawled_at': datetime.now().isoformat(),
                'error_message': error_message,
                'response_code': str(response_code) if response_code else ''
            })
            self._urls_cache_modified = True
            self._save_urls_cache()

    def update_url_retry_count(self, url: str):
        """Increment retry count for URL"""
        self._load_urls_cache()
        normalized_url = self._normalize_url(url)

        if normalized_url in self._urls_cache:
            current_count = int(self._urls_cache[normalized_url]['retry_count'])
            self._urls_cache[normalized_url]['retry_count'] = str(current_count + 1)
            self._urls_cache_modified = True
            self._save_urls_cache()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive crawling statistics"""
        self._load_urls_cache()

        status_counts = {'discovered': 0, 'queued': 0, 'crawled': 0, 'failed': 0}
        files_saved = 0
        total_retries = 0
        rate_limited = 0

        for row in self._urls_cache.values():
            status = row['status']
            if status in status_counts:
                status_counts[status] += 1

            if row['file_path']:
                files_saved += 1

            total_retries += int(row['retry_count'])

            if row['response_code'] == '429':
                rate_limited += 1

        return {
            'discovered': status_counts['discovered'],
            'queued': status_counts['queued'],
            'crawled': status_counts['crawled'],
            'failed': status_counts['failed'],
            'files_saved': files_saved,
            'total_urls': sum(status_counts.values()),
            'total_retries': total_retries,
            'rate_limited': rate_limited
        }

    def recover_interrupted_crawl(self, max_retries: int) -> int:
        """Reset URLs from interrupted crawls"""
        self._load_urls_cache()
        reset_count = 0

        for normalized_url, row in self._urls_cache.items():
            if (row['status'] == 'queued' and not row['crawled_at'] and
                    int(row['retry_count']) < max_retries):
                row['status'] = 'discovered'
                reset_count += 1
            elif (row['status'] == 'failed' and not row['crawled_at'] and
                  int(row['retry_count']) < max_retries):
                row['status'] = 'discovered'
                reset_count += 1

        if reset_count > 0:
            self._urls_cache_modified = True
            self._save_urls_cache()

        return reset_count


class CSVCrawler:
    def __init__(self, data_dir='rotten_tomatoes_data', csv_dir='csv_data'):
        self.data_dir = data_dir
        self.csv_dir = csv_dir

        # Initialize CSV manager
        self.csv_manager = CSVCrawlerManager(csv_dir)

        self.config = {
            'USER_AGENTS': [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            ],
            'REQUEST_DELAY': (2, 5),
            'REQUESTS_PER_MINUTE': 30,
            'REQUESTS_PER_HOUR': 1500,
            'MAX_RETRIES': 1,
            'TIMEOUT': 25,
            'MAX_URLS_PER_RUN': 1000000,
            'CRAWL_DELAY': 3,
            'BATCH_SIZE': 50,
            'CHECKPOINT_INTERVAL': 100,
            'MAX_DEPTH': 10,
            'USER_AGENT_ROTATION_INTERVAL': 10,
        }

        # Create directories
        self.html_dir = os.path.join(data_dir, 'html_pages')
        os.makedirs(self.html_dir, exist_ok=True)

        # Robots.txt parsers cache
        self.robot_parsers = {}

        # Request tracking for rate limiting
        self.minute_requests = deque(maxlen=self.config['REQUESTS_PER_MINUTE'] * 2)
        self.hour_requests = deque(maxlen=self.config['REQUESTS_PER_HOUR'] * 2)

        # Anti-ban tracking
        self.consecutive_errors = 0
        self.last_request_time = 0
        self.request_count = 0

        # Session and configuration
        self.session = requests.Session()
        self._setup_session()

    def _setup_session(self):
        """Setup requests session with proper headers and retry strategy"""
        self.session.headers.update({
            'User-Agent': random.choice(self.config['USER_AGENTS']),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
        })

        # Setup retry strategy
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _get_robots_parser(self, domain):
        """Get or create robots.txt parser for a domain with caching"""
        if domain in self.robot_parsers:
            return self.robot_parsers[domain]

        try:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = f"https://{domain}/robots.txt"

            # Rate limit before fetching robots.txt
            self._rate_limit()

            response = self.session.get(robots_url, timeout=self.config['TIMEOUT'])
            if response.status_code == 200:
                rp.parse(response.text.splitlines())
                self.robot_parsers[domain] = rp
                print(f"Loaded robots.txt for {domain}")
                return rp
            else:
                # If no robots.txt, create a permissive parser
                print(f"No robots.txt found for {domain} (status: {response.status_code})")
                rp = urllib.robotparser.RobotFileParser()
                rp.allow_all = True
                self.robot_parsers[domain] = rp
                return rp

        except Exception as e:
            print(f"Error loading robots.txt for {domain}: {e}")
            # Create permissive parser on error
            rp = urllib.robotparser.RobotFileParser()
            rp.allow_all = True
            self.robot_parsers[domain] = rp
            return rp

    def _is_allowed_by_robots(self, url, user_agent=None):
        """Check if URL is allowed by robots.txt"""
        if user_agent is None:
            user_agent = self.session.headers['User-Agent']

        parsed = urlparse(url)
        domain = parsed.netloc

        if not domain:
            return False

        parser = self._get_robots_parser(domain)
        return parser.can_fetch(user_agent, url)

    def _get_robots_crawl_delay(self, domain, user_agent=None):
        """Get crawl delay from robots.txt"""
        if user_agent is None:
            user_agent = self.session.headers['User-Agent']

        parser = self._get_robots_parser(domain)
        return parser.crawl_delay(user_agent) or self.config['CRAWL_DELAY']

    def _get_random_user_agent(self):
        return random.choice(self.config['USER_AGENTS'])

    def _rotate_user_agent(self):
        """Rotate user agent to avoid detection"""
        new_agent = self._get_random_user_agent()
        self.session.headers['User-Agent'] = new_agent
        print(f"Rotated user agent to: {new_agent[:50]}...")

    def _get_robots_compliant_delay(self, url):
        """Get delay that respects robots.txt and our config with randomization"""
        parsed = urlparse(url)
        domain = parsed.netloc

        robots_delay = self._get_robots_crawl_delay(domain)
        config_delay = random.uniform(*self.config['REQUEST_DELAY'])

        # Add some randomness to avoid patterns
        jitter = random.uniform(0.5, 1.5)
        final_delay = max(robots_delay, config_delay) * jitter

        return final_delay

    def _rate_limit(self):
        """Implement intelligent rate limiting with error backoff"""
        now = time.time()

        # Clean old requests
        self.minute_requests = deque(
            (r for r in self.minute_requests if now - r < 60),
            maxlen=self.config['REQUESTS_PER_MINUTE'] * 2
        )
        self.hour_requests = deque(
            (r for r in self.hour_requests if now - r < 3600),
            maxlen=self.config['REQUESTS_PER_HOUR'] * 2
        )

        # Only apply error backoff for actual blocking errors (not 404s)
        # Use a smaller error factor since we're handling 404s separately
        error_factor = 1 + (max(0, self.consecutive_errors - 2) * 0.3)
        effective_minute_limit = max(10, self.config['REQUESTS_PER_MINUTE'] / error_factor)
        effective_hour_limit = max(100, self.config['REQUESTS_PER_HOUR'] / error_factor)

        # Check minute limit
        if len(self.minute_requests) >= effective_minute_limit:
            sleep_time = 60 - (now - self.minute_requests[0])
            if sleep_time > 0:
                print(f"Minute rate limit: Sleeping {sleep_time:.1f}s (consecutive errors: {self.consecutive_errors})")
                time.sleep(sleep_time)

        # Check hour limit
        if len(self.hour_requests) >= effective_hour_limit:
            sleep_time = 3600 - (now - self.hour_requests[0])
            if sleep_time > 0:
                print(f"Hourly rate limit: Sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        now = time.time()
        self.minute_requests.append(now)
        self.hour_requests.append(now)

    def _normalize_url(self, url):
        """Normalize URL to avoid duplicates"""
        parsed = urlparse(url)

        # Standardize scheme and netloc
        scheme = 'https'
        netloc = parsed.netloc.lower()

        if netloc.startswith('www.'):
            netloc = netloc[4:]

        # Normalize path - remove trailing slashes
        path = parsed.path.rstrip('/')
        if not path:
            path = '/'

        # Remove common tracking parameters
        query_params = []
        if parsed.query:
            for param in parsed.query.split('&'):
                if not any(track_param in param.lower() for track_param in
                           ['utm_', 'ref=', 'source=', 'campaign=', 'medium=']):
                    query_params.append(param)

        query = '&'.join(query_params) if query_params else ''

        # Reconstruct URL
        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"

        return normalized

    def _url_to_filename(self, url):
        """Convert URL to safe filename with hash"""
        clean_url = re.sub(r'^https?://', '', url)
        clean_url = re.sub(r'[^a-zA-Z0-9]', '_', clean_url)

        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]

        if len(clean_url) > 100:
            clean_url = clean_url[:100]

        return f"{clean_url}_{url_hash}.html"

    def _is_valid_url(self, url):
        """Check if URL is valid for crawling"""
        parsed = urlparse(url)

        # Must be from Rotten Tomatoes
        if parsed.netloc not in ['www.rottentomatoes.com', 'rottentomatoes.com']:
            return False

        # Check robots.txt compliance
        if not self._is_allowed_by_robots(url):
            print(f"Blocked by robots.txt: {url}")
            return False

        # Avoid problematic paths
        excluded_paths = [
            '/user/', '/api/', '/search', '/login', '/signup',
            '/edit/', '/submit', '/account/', '/cart/', '/checkout/',
            '/auth/', '/oauth/', '/admin/', '/static/', '/assets/', '/videos', '/pictures'
        ]
        if any(path in parsed.path for path in excluded_paths):
            return False

        # Focus on content pages
        # allowed_patterns = [
        #     '/m/', '/tv/', '/browse/', '/news/', '/critics/',
        #     '/celebrity/', '/person/', '/series/', '/season/',
        #     '/episode/', '/article/', '/interview/', '/review/'
        # ]
        # changed to check only movies to get more meaningful results
        allowed_patterns = ['/m/']

        # Allow specific file extensions
        allowed_extensions = ['.html', '.htm', '.php', '.jsp', '.asp', '']
        path_ext = os.path.splitext(parsed.path)[1].lower()

        if any(pattern in parsed.path for pattern in allowed_patterns) and path_ext in allowed_extensions:
            return True

        # Allow homepage and main sections
        # if parsed.path in ['/', '/browse', '/tv', '/news', '/movies', '/tv']:
        #    return True

        return False

    def _extract_links(self, html_content, base_url):
        """Extract all valid links from HTML content using improved regex"""
        links = set()

        # Multiple regex patterns to catch different link formats
        patterns = [
            r'<a\s+[^>]*href=(["\'])(.*?)\1',  # Standard href
            r'href=(["\'])(.*?)\1',  # Just href attribute
            r'data-href=(["\'])(.*?)\1',  # data-href attributes
            r'data-url=(["\'])(.*?)\1',  # data-url attributes
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, html_content, re.IGNORECASE):
                href = match.group(2)

                # Skip unwanted links
                if href.startswith(('javascript:', 'mailto:', '#', 'tel:')):
                    continue
                if href.endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js')):
                    continue

                # Convert relative URLs to absolute
                full_url = urljoin(base_url, href)

                # Normalize URL
                normalized_url = self._normalize_url(full_url)

                if self._is_valid_url(normalized_url):
                    links.add(normalized_url)

        return list(links)

    def _recover_interrupted_crawl(self):
        """Comprehensive recovery for interrupted crawls"""
        recovered = self.csv_manager.recover_interrupted_crawl(self.config['MAX_RETRIES'])
        if recovered > 0:
            print(f"Resume: Reset {recovered} URLs from previous session")
        return recovered

    def _create_checkpoint(self, total_crawled):
        """Create recovery checkpoint"""
        stats = self.get_stats()
        checkpoint_data = {
            'id': '1',
            'checkpoint_time': datetime.now().isoformat(),
            'urls_crawled': str(total_crawled),
            'urls_queued': str(stats['discovered']),
            'total_urls': str(stats['total_urls'])
        }

        # Write checkpoint to CSV
        checkpoint_file = os.path.join(self.csv_dir, 'crawl_checkpoints.csv')
        with open(checkpoint_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'checkpoint_time', 'urls_crawled', 'urls_queued', 'total_urls']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(checkpoint_data)

        print(f"Checkpoint saved: {total_crawled} URLs crawled, {stats['discovered']} in queue")

    def _get_folder_size(self):
        """Get total size of crawled data in MB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.html_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB

    def _estimate_completion_time(self, current_count, target_count):
        """Estimate time to completion"""
        if current_count == 0:
            return "Unknown"

        # Calculate average time per URL
        avg_time_per_url = (self.request_count * 3) / current_count if current_count > 0 else 5

        remaining_urls = target_count - current_count
        remaining_seconds = remaining_urls * avg_time_per_url

        # Convert to readable format
        if remaining_seconds < 3600:
            return f"{remaining_seconds / 60:.1f} minutes"
        elif remaining_seconds < 86400:
            return f"{remaining_seconds / 3600:.1f} hours"
        else:
            return f"{remaining_seconds / 86400:.1f} days"

    def crawl_url(self, url, depth):
        """Crawl a single URL and return discovered links with enhanced error handling"""
        # Final robots.txt check before crawling
        if not self._is_allowed_by_robots(url):
            print(f"Robots.txt blocked: {url}")
            self.csv_manager.mark_url_failed(url, "Blocked by robots.txt")
            return []

        for attempt in range(self.config['MAX_RETRIES'] + 1):
            try:
                self._rate_limit()

                # Rotate user agent periodically
                if self.request_count % self.config['USER_AGENT_ROTATION_INTERVAL'] == 0:
                    self._rotate_user_agent()

                delay = self._get_robots_compliant_delay(url)
                time.sleep(delay)

                print(f"[Depth {depth}] Crawling: {url}")
                response = self.session.get(url, timeout=self.config['TIMEOUT'])

                # Handle different status codes appropriately
                if response.status_code == 404:  # Not Found - not a blocking issue
                    print(f"Page not found (404): {url}")
                    self.csv_manager.mark_url_failed(url, "Page not found", 404)
                    return []  # Return empty list, don't count as error for rate limiting

                if response.status_code == 429:  # Too Many Requests
                    self.consecutive_errors += 1
                    backoff_time = (2 ** attempt) * 60  # Exponential backoff in minutes
                    print(f"Rate limited! Backing off for {backoff_time}s")
                    time.sleep(backoff_time)
                    continue

                if response.status_code == 403:  # Forbidden
                    print(f"Access forbidden - possible IP ban: {url}")
                    self.csv_manager.mark_url_failed(url, "Access Forbidden", 403)
                    self.consecutive_errors += 1
                    return []

                # Handle other 4xx client errors (except 404, 429, 403 which we re handling above)
                if 400 <= response.status_code < 500:
                    print(f"Client error {response.status_code}: {url}")
                    self.csv_manager.mark_url_failed(url, f"Client error {response.status_code}", response.status_code)
                    return []  # Don't retry client errors (except 429)

                response.raise_for_status()

                # Reset error counter on success
                self.consecutive_errors = max(0, self.consecutive_errors - 1)

                # Validate content type
                content_type = response.headers.get('content-type', '')
                if 'text/html' not in content_type:
                    print(f"Skipping non-HTML content: {content_type}")
                    self.csv_manager.mark_url_failed(url, f"Non-HTML content: {content_type}", response.status_code)
                    return []

                # Check for blocking pages (captcha, etc.)
                if any(indicator in response.text.lower() for indicator in
                       ['captcha', 'access denied', 'bot detected', 'cloudflare']):
                    print(f"Blocking page detected: {url}")
                    self.csv_manager.mark_url_failed(url, "Blocking page detected", response.status_code)
                    self.consecutive_errors += 1
                    return []

                # Save HTML to file
                filename = self._url_to_filename(url)
                file_path = os.path.join(self.html_dir, filename)

                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                except IOError as e:
                    print(f"Failed to write {file_path}: {e}")
                    file_path = None

                # Extract links from this page
                new_links = self._extract_links(response.text, url)

                # Mark as crawled
                self.csv_manager.mark_url_crawled(url, file_path, response.status_code)

                self.request_count += 1
                print(f"Saved {filename} - Found {len(new_links)} links")
                return new_links

            except requests.exceptions.RequestException as e:
                # Don't count 404s as consecutive errors for rate limiting
                if hasattr(e, 'response') and e.response is not None:
                    if e.response.status_code == 404:
                        print(f"Page not found (404): {url}")
                        self.csv_manager.mark_url_failed(url, "Page not found", 404)
                        return []
                    elif 400 <= e.response.status_code < 500:
                        print(f"Client error {e.response.status_code}: {url}")
                        self.csv_manager.mark_url_failed(url, f"Client error {e.response.status_code}",
                                                         e.response.status_code)
                        return []

                # Only count network errors and server errors as consecutive errors
                self.consecutive_errors += 1
                print(f"Attempt {attempt + 1} failed for {url}: {e}")

                # Update retry count
                self.csv_manager.update_url_retry_count(url)

                if attempt < self.config['MAX_RETRIES']:
                    retry_delay = (2 ** attempt) * 30  # Exponential backoff
                    print(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"All attempts failed for {url}")
                    self.csv_manager.mark_url_failed(url, str(e))
                    return []

            except Exception as e:
                self.consecutive_errors += 1
                print(f"Unexpected error crawling {url}: {e}")
                self.csv_manager.mark_url_failed(url, f"Unexpected error: {str(e)}")
                return []

        return []

    def start_crawling(self, start_urls=None, max_urls=None):
        """Start the crawling process with enhanced resume capability"""
        if start_urls is None:
            start_urls = ['https://www.rottentomatoes.com/']

        if max_urls is None:
            max_urls = self.config['MAX_URLS_PER_RUN']

        print(f"Starting crawl session - Target: {max_urls} URLs")

        # Recover from interrupted crawl
        recovered = self._recover_interrupted_crawl()
        if recovered > 0:
            print(f"Recovered {recovered} URLs from previous session")

        # Pre-fetch robots.txt for main domain
        print("Loading robots.txt...")
        self._get_robots_parser('www.rottentomatoes.com')

        # Add starting URLs to database
        print("Adding starting URLs to database...")
        self.csv_manager.add_urls(start_urls, depth=0)

        total_crawled = 0
        batch_count = 0
        consecutive_empty_batches = 0

        start_time = time.time()

        while total_crawled < max_urls and consecutive_empty_batches < 10:
            # Get next batch of URLs
            next_urls = self.csv_manager.get_next_urls(limit=self.config['BATCH_SIZE'])

            if not next_urls:
                consecutive_empty_batches += 1
                print(f"No URLs found in queue, waiting... ({consecutive_empty_batches}/10)")
                time.sleep(30)
                continue

            consecutive_empty_batches = 0
            batch_new_links = []

            for url, depth in next_urls:
                if total_crawled >= max_urls:
                    break

                # Crawl URL
                new_links = self.crawl_url(url, depth)
                batch_new_links.extend(new_links)

                total_crawled += 1

                # Progress reporting
                if total_crawled % 10 == 0:
                    self._log_progress(total_crawled, max_urls, start_time)

                # Rotate user agent every N requests
                if total_crawled % self.config['USER_AGENT_ROTATION_INTERVAL'] == 0:
                    self._rotate_user_agent()

                # Checkpoint and extended break every N URLs
                if total_crawled % self.config['CHECKPOINT_INTERVAL'] == 0 and total_crawled < max_urls:
                    self._create_checkpoint(total_crawled)

                    # Extended break to avoid detection
                    long_break = random.uniform(60, 180)
                    print(f"Taking extended break: {long_break:.1f}s...")
                    time.sleep(long_break)

            # Add new links to database
            if batch_new_links:
                new_count = self.csv_manager.add_urls(batch_new_links, depth + 1)
                print(f"Added {new_count} new URLs to database")

            batch_count += 1

            # Memory cleanup for long runs
            if batch_count % 20 == 0:
                self._cleanup_memory()

        # Final checkpoint
        self._create_checkpoint(total_crawled)

        total_size_mb = self._get_folder_size()
        elapsed_time = time.time() - start_time

        print(f"\nCrawling session completed!")
        print(f"Total URLs crawled: {total_crawled}")
        print(f"Total data collected: {total_size_mb:.2f} MB")
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print(f"Average speed: {total_crawled / (elapsed_time / 3600):.1f} URLs/hour")

        return total_crawled

    def _log_progress(self, current, total, start_time):
        """Detailed progress logging"""
        stats = self.get_stats()
        elapsed = time.time() - start_time
        progress_pct = (current / total) * 100

        # Calculate speed
        urls_per_hour = (current / elapsed) * 3600 if elapsed > 0 else 0

        # Estimate completion
        eta = self._estimate_completion_time(current, total)

        print(f"\nProgress: {current:,}/{total:,} ({progress_pct:.1f}%)")
        print(f"   Speed: {urls_per_hour:.1f} URLs/hour | ETA: {eta}")
        print(f"   Discovered: {stats['discovered']:,} | Queued: {stats['queued']:,}")
        print(f"   Crawled: {stats['crawled']:,} | Failed: {stats['failed']:,}")
        print(f"   Consecutive errors: {self.consecutive_errors}")

    def _cleanup_memory(self):
        """Clean up memory for long-running processes"""
        import gc
        gc.collect()

        # Reset session periodically to prevent memory leaks
        if hasattr(self, 'session'):
            self.session.close()
            self._setup_session()

        print("Memory cleanup completed")

    def get_stats(self):
        """Get comprehensive crawling statistics"""
        stats = self.csv_manager.get_stats()
        stats.update({
            'consecutive_errors': self.consecutive_errors,
            'request_count': self.request_count
        })
        return stats

    def get_resume_info(self):
        """Get information for resuming interrupted crawls"""
        stats = self.get_stats()

        print("\nResume Information:")
        print(f"   Current discovered: {stats['discovered']:,} URLs")
        print(f"   Current crawled: {stats['crawled']:,} URLs")
        print(f"   Current total: {stats['total_urls']:,} URLs")

        return stats

    def close(self):
        """Close database connection and cleanup"""
        if hasattr(self, 'session'):
            self.session.close()
        print("Crawler shutdown complete")


def signal_handler(signum, frame):
    """Handle graceful shutdown"""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    if 'crawler' in globals():
        crawler.close()
    sys.exit(0)


def main():
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    global crawler
    crawler = CSVCrawler(
        data_dir='rotten_tomatoes_data',
        csv_dir='csv_data'
    )

    try:
        print("Rotten Tomatoes CSV Crawler")
        print("======================================")

        # Show initial stats and resume info
        crawler.get_resume_info()
        stats = crawler.get_stats()
        print(f"\nInitial state:")
        print(f"Total URLs: {stats['total_urls']:,}")
        print(f"Crawled: {stats['crawled']:,}, Discovered: {stats['discovered']:,}")

        # Start crawling
        start_urls = [
            'https://www.rottentomatoes.com/'
        ]

        print(f"\nStarting crawl from {len(start_urls)} seed URLs...")
        print(f"Press Ctrl+C to stop gracefully\n")

        pages_crawled = crawler.start_crawling(start_urls=start_urls, max_urls=1000000)

        # Final stats
        stats = crawler.get_stats()
        print(f"\nFinal Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:,}")

    except Exception as e:
        print(f"Crawler error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        crawler.close()


if __name__ == "__main__":
    main()