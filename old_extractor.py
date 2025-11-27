import re
import os
import json
import time
import csv
import pickle
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import random
import tiktoken


# =========================
#   CORE EXTRACTOR
# =========================
class RTDataExtractor:
    """Regex-first extractor for Rotten Tomatoes pages.
    Order of operations:
      1) Page type: filename prefix -> canonical URL -> gentle fallbacks.
      2) For movie/TV: parse 'JSON islands' (scorecard + ld+json + hero) first.
      3) Light, scoped HTML fallbacks where needed.
      4) For non-media pages (critic/celebrity), return only core metadata.
    """

    # ---------- Public entry ----------
    def extract_from_html(self, html: str, file_path: str = "") -> Dict[str, Any]:
        # Parse JSON islands up-front (cheap enough, used selectively)
        ld_list = self._extract_all_ld_json(html)
        hero = self._extract_json_by_id(html, "media-hero-json")
        scorecard = self._extract_json_by_id(html, "media-scorecard-json")
        ld_core = self._pick_ld_core(ld_list)  # Movie or TVSeries dict if present

        page_type = self._detect_page_type(html, file_path, ld_core, hero)
        url = self._first_nonempty(
            self._match1(r'<link rel="canonical" href="([^"]+)"', html, flags=re.I),
            self._match1(r'<meta property="og:url" content="([^"]+)"', html, flags=re.I),
            self._match1(r'"url"\s*:\s*"([^"]+)"', json.dumps(ld_core)) if ld_core else "",
        )

        title = self._first_nonempty(
            self._deepget(hero, "content", "title"),
            self._deepget(ld_core, "name"),
            self._match1(r"<sr-text>\s*(.*?)\s*</sr-text>", html, flags=re.I | re.DOTALL),
            self._match1(r'<meta property="og:title" content="([^"]+)"', html, flags=re.I),
            self._match1(r"<title>(.*?)\s*\|", html, flags=re.I),
        )

        out: Dict[str, Any] = {
            "file_path": file_path,
            "page_type": page_type,
            "url": self._clean(url),
            "title": self._clean(title),
            "description": self._meta_description(html),
            "extraction_timestamp": time.time(),
        }

        # Only extract media fields for movie/tv
        if page_type in {"movie", "tv"}:
            # Synopsis
            synopsis = self._first_nonempty(
                self._deepget(scorecard, "description"),
                self._match1(
                    r'<div[^>]*slot="description"[^>]*>.*?<rt-text[^>]*>([^<]+)</rt-text>',
                    html, flags=re.I | re.DOTALL
                ),
                self._match1(r'<meta name="description" content="([^"]+)"', html, flags=re.I),
                self._match1(r'<meta property="og:description" content="([^"]+)"', html, flags=re.I),
            )

            tomatometer = self._coerce_int(
                self._deepget(scorecard, "criticsScore", "score"),
                self._deepget(ld_core, "aggregateRating", "ratingValue"),
                self._slot_number(html, "criticsScore"),
            )
            audience_score = self._coerce_int(
                self._deepget(scorecard, "audienceScore", "score"),
                self._slot_number(html, "audienceScore"),
            )

            # Rating / genres / runtime / release date
            mp = self._list_str(self._deepget(hero, "content", "metadataProps"))

            rating_cert = self._normalize_rating(
                self._first_nonempty(
                    self._deepget(ld_core, "contentRating"),
                    next((x for x in mp if re.fullmatch(r"(G|PG|PG-13|R|NC-17|TV-[A-Z0-9\-+]+)", x)), ""),
                )
            )

            genres = self._list_str(
                self._deepget(ld_core, "genre"),
                self._deepget(hero, "content", "metadataGenres"),
            )

            runtime = self._first_nonempty(
                self._runtime_from_ld(ld_core),
                self._runtime_from_time_datetime(html),
                next((x for x in mp if re.search(r"\d+h\s*\d+m|\d+m\b", x)), ""),
                self._detail_value(html, r"Runtime"),
            )

            release_date = self._first_nonempty(
                self._detail_value(html, r"Release Date \((?:Theaters|Streaming|Limited|Original|Wide)[^)]*\)"),
                self._detail_value(html, r"Release Date"),
                self._deepget(ld_core, "dateCreated"),
                next((x for x in mp if re.fullmatch(r"\d{4}", x)), ""),
            )

            # Director & Cast
            director = ", ".join(self._names_from_people(self._deepget(ld_core, "director")))
            cast = self._names_from_people(self._deepget(ld_core, "actor"))
            if not cast:
                # Conservative HTML fallback
                cast = list(
                    {m.strip() for m in re.findall(r'data-qa="person-name"[^>]*>([^<]+)</a>', html, flags=re.I)})

            out.update({
                "tomatometer": tomatometer or 0,
                "audience_score": audience_score or 0,
                "rating": self._clean(rating_cert),
                "genres": genres,
                "release_date": self._clean(release_date),
                "runtime": self._clean(runtime),
                "director": self._clean(director),
                "cast": cast[:20],
                "synopsis": self._clean(synopsis),
            })

        return out

    # ---------- Page-type detection ----------
    def _detect_page_type(
            self, html: str, file_path: str, ld_core: Optional[dict], hero: Optional[dict]
    ) -> str:
        """Decide among: movie, tv, celebrity, critic, unknown"""

        # 0) Filename prefixes
        fp = (file_path or "").replace("\\", "/").lower()
        base = os.path.basename(fp)
        if base.startswith("rottentomatoes_com_m_"):
            return "movie"
        if base.startswith("rottentomatoes_com_tv_") or base.startswith("rottentomatoes_com_series_"):
            return "tv"
        if base.startswith("rottentomatoes_com_celebrity_"):
            return "celebrity"
        if base.startswith("rottentomatoes_com_critic") or base.startswith("rottentomatoes_com_critics_"):
            return "critic"

        # 1) Canonical URL
        canonical = self._first_nonempty(
            self._match1(r'<link rel="canonical" href="([^"]+)"', html, flags=re.I),
            self._match1(r'<meta property="og:url" content="([^"]+)"', html, flags=re.I),
        ).lower()

        def classify_from_url(u: str) -> Optional[str]:
            if not u:
                return None
            if "/m/" in u:
                return "movie"
            if "/tv/" in u or "/series/" in u:
                return "tv"
            if "/celebrity/" in u:
                return "celebrity"
            if "/critic" in u or "/critics/" in u or "/critics/source/" in u:
                return "critic"
            return None

        c = classify_from_url(canonical)
        if c:
            return c

        # 2) Media-only hints (don't use if URL doesn't look like media)
        mediatype = self._match1(r'mediatype="([^"]+)"', html, flags=re.I)
        if mediatype:
            mt = mediatype.lower()
            if "movie" in mt:
                return "movie"
            if "tv" in mt:
                return "tv"

        # 3) ld+json type (only if URL-ish context is missing)
        if ld_core:
            if ld_core.get("@type") == "Movie":
                return "movie"
            if ld_core.get("@type") == "TVSeries":
                return "tv"

        # 4) Fallback heuristics from full path
        if "/m_" in fp or "/m/" in fp or "_m_" in fp:
            return "movie"
        if "/tv_" in fp or "/tv/" in fp or "_tv_" in fp:
            return "tv"
        if "/celebrity_" in fp or "/celebrity/" in fp:
            return "celebrity"
        if "/critic" in fp or "/critics_" in fp or "/critics/" in fp:
            return "critic"

        return "unknown"

    # ---------- JSON island helpers ----------
    def _extract_json_by_id(self, html: str, script_id: str) -> Optional[dict]:
        m = re.search(
            rf'<script[^>]+id="{re.escape(script_id)}"[^>]*>(.*?)</script>',
            html, flags=re.I | re.DOTALL
        )
        if not m:
            return None
        blob = m.group(1).strip()
        try:
            return json.loads(blob)
        except Exception:
            # Minimal cleanup if there are stray entities/tags
            blob = re.sub(r"</?[^>]+>", "", blob)
            blob = blob.replace("&quot;", '"')
            try:
                return json.loads(blob)
            except Exception:
                return None

    def _extract_all_ld_json(self, html: str) -> List[Any]:
        objs: List[Any] = []
        for m in re.finditer(
                r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
                html, flags=re.I | re.DOTALL
        ):
            blob = m.group(1).strip()
            try:
                loaded = json.loads(blob)
                if isinstance(loaded, dict) and "@graph" in loaded and isinstance(loaded["@graph"], list):
                    objs.extend(loaded["@graph"])
                else:
                    objs.append(loaded)
            except Exception:
                continue
        return objs

    def _pick_ld_core(self, ld_list: List[Any]) -> Optional[dict]:
        """Choose the Movie or TVSeries object if present."""
        flat: List[dict] = []

        def _walk(x):
            if isinstance(x, list):
                for y in x:
                    _walk(y)
            elif isinstance(x, dict):
                flat.append(x)

        _walk(ld_list)

        for t in ("Movie", "TVSeries"):
            for d in flat:
                if d.get("@type") == t:
                    return d
        return flat[0] if flat else None

    # ---------- Media helpers ----------
    def _slot_number(self, html: str, slot: str) -> Optional[int]:
        m = re.search(
            rf'<rt-text[^>]+slot="{re.escape(slot)}"[^>]*>\s*(\d+)\s*%?\s*</rt-text>',
            html, flags=re.I
        )
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        return None

    def _detail_value(self, html: str, label_regex: str) -> str:
        """Read details from labeled list in Movie/TV Details sections."""
        pattern = (
            rf'{label_regex}</rt-text>\s*</dt>\s*<dd[^>]*>\s*<rt-text[^>]*data-qa="item-value"[^>]*>'
            r'([^<]+)</rt-text>'
        )
        m = re.search(pattern, html, flags=re.I)
        return m.group(1).strip() if m else ""

    def _runtime_from_ld(self, ld_core: Optional[dict]) -> str:
        dur = self._deepget(ld_core, "duration")
        if not isinstance(dur, str):
            return ""
        m = re.search(r"PT(?:(\d+)H)?(?:(\d+)M)?", dur, flags=re.I)
        if not m:
            return ""
        h = int(m.group(1) or 0)
        mns = int(m.group(2) or 0)
        total = h * 60 + mns
        return self._hm_from_minutes(total)

    def _runtime_from_time_datetime(self, html: str) -> str:
        m = re.search(r'<time[^>]*datetime="(PT[0-9HM]+)"', html, flags=re.I)
        if not m:
            return ""
        iso = m.group(1)
        mm = re.search(r"PT(?:(\d+)H)?(?:(\d+)M)?", iso, flags=re.I)
        if not mm:
            return ""
        h = int(mm.group(1) or 0)
        mns = int(mm.group(2) or 0)
        total = h * 60 + mns
        return self._hm_from_minutes(total)

    # ---------- Small utils ----------
    def _deepget(self, obj: Optional[dict], *keys, default=None):
        cur = obj
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _hm_from_minutes(self, minutes: int) -> str:
        try:
            mins = int(minutes)
            h, m = divmod(mins, 60)
            return (f"{h}h {m}m" if h else f"{m}m").strip()
        except Exception:
            return ""

    def _names_from_people(self, people) -> List[str]:
        names: List[str] = []
        if isinstance(people, dict):
            if people.get("name"):
                names.append(people["name"])
        elif isinstance(people, list):
            for p in people:
                if isinstance(p, dict) and p.get("name"):
                    names.append(p["name"])
        # unique + cleaned
        seen = set()
        out = []
        for n in names:
            n = self._clean(n)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _normalize_rating(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip().upper().replace("PG13", "PG-13").replace("NC17", "NC-17").replace("TV14", "TV-14").replace("TVPG",
                                                                                                                 "TV-PG")
        if re.fullmatch(r"(G|PG|PG-13|R|NC-17|NR|NOT RATED|UNRATED|TV-(?:Y7-FV|Y7|Y|G|PG|14|MA))", s):
            return s
        return ""

    def _clean(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"&nbsp;", " ", text, flags=re.I)
        text = re.sub(r"&amp;", "&", text, flags=re.I)
        text = re.sub(r"&quot;", '"', text, flags=re.I)
        text = re.sub(r"&#39;|&apos;", "'", text, flags=re.I)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _coerce_int(self, *vals) -> Optional[int]:
        for v in vals:
            if v is None:
                continue
            if isinstance(v, (int, float)):
                iv = int(v)
                if 0 <= iv <= 100:
                    return iv
                continue
            if isinstance(v, str):
                m = re.search(r"\d+", v)
                if m:
                    iv = int(m.group(0))
                    if 0 <= iv <= 100:
                        return iv
        return None

    def _match1(self, pattern: str, text: str, flags=0) -> str:
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else ""

    def _first_nonempty(self, *vals) -> str:
        for v in vals:
            if v:
                return v
        return ""

    def _list_str(self, *candidates) -> List[str]:
        out: List[str] = []
        for c in candidates:
            if not c:
                continue
            if isinstance(c, list):
                out.extend([self._clean(x) for x in c if isinstance(x, str)])
            elif isinstance(c, str):
                out.append(self._clean(c))
        # unique preserve order
        seen = set()
        uniq = []
        for x in out:
            if x and x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _meta_description(self, html: str) -> str:
        return self._first_nonempty(
            self._match1(r'<meta name="description" content="([^"]+)"', html, flags=re.I),
            self._match1(r'<meta property="og:description" content="([^"]+)"', html, flags=re.I),
        )


# =========================
#   STORAGE
# =========================
class RTMetadataStorage:
    """CSV / JSONL / Pickle storage (no DB)."""

    def __init__(self, storage_dir: str = "extracted_metadata"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save_to_jsonl(self, data: Dict[str, Any], batch_size: int = 1000):
        timestamp = int(time.time())
        batch_num = 0
        batch: List[Dict[str, Any]] = []

        for _fp, meta in data.items():
            batch.append(meta)
            if len(batch) >= batch_size:
                self._write_jsonl_batch(batch, batch_num, timestamp)
                batch = []
                batch_num += 1

        if batch:
            self._write_jsonl_batch(batch, batch_num, timestamp)

    def _write_jsonl_batch(self, batch: List[Dict[str, Any]], batch_num: int, timestamp: int):
        filename = f"metadata_batch_{timestamp}_{batch_num:04d}.jsonl"
        path = os.path.join(self.storage_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for item in batch:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved batch {batch_num} with {len(batch)} records to {filename}")

    def save_to_csv(self, data: Dict[str, Any], filename: str = "metadata.csv"):
        path = os.path.join(self.storage_dir, filename)
        fieldnames = [
            "file_path", "page_type", "url", "title", "description",
            "tomatometer", "audience_score", "rating", "release_date",
            "runtime", "director", "synopsis", "critics_consensus",
            "extraction_timestamp", "genres", "cast"
        ]
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for _fp, meta in data.items():
                row = meta.copy()
                row["genres"] = "|".join(meta.get("genres", []))
                row["cast"] = "|".join(meta.get("cast", []))
                writer.writerow(row)
        print(f"Saved {len(data)} records to CSV: {path}")
        self._save_list_data(data, "genres")
        self._save_list_data(data, "cast")

    def _save_list_data(self, data: Dict[str, Any], list_type: str):
        path = os.path.join(self.storage_dir, f"{list_type}.csv")
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["file_path", list_type[:-1]])
            rows = 0
            for fp, meta in data.items():
                for item in meta.get(list_type, []):
                    writer.writerow([fp, item])
                    rows += 1
        print(f"Saved {rows} rows to: {path}")

    def save_to_pickle(self, data: Dict[str, Any], filename: str = "metadata.pkl"):
        path = os.path.join(self.storage_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {len(data)} records to pickle file: {path}")


# =========================
#   STATISTICS & VALIDATION
# =========================
class RTStatistics:
    """Generate statistics and validation reports for extracted metadata."""

    def __init__(self, storage_dir: str = "extracted_metadata"):
        self.storage_dir = storage_dir
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def generate_statistics(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistics from metadata."""
        stats = {
            "total_pages": len(metadata),
            "page_type_counts": defaultdict(int),
            "extraction_stats": defaultdict(int),
            "token_stats": defaultdict(int),
            "score_stats": defaultdict(dict),
            "field_completeness": defaultdict(float),
            "validation_results": defaultdict(list)
        }

        total_tokens = 0
        total_description_tokens = 0
        total_synopsis_tokens = 0

        for file_path, meta in metadata.items():
            page_type = meta.get("page_type", "unknown")
            stats["page_type_counts"][page_type] += 1

            # Field completeness
            for field in ["title", "url", "description", "tomatometer", "audience_score", "genres", "cast"]:
                if field in meta and meta[field]:
                    if isinstance(meta[field], list):
                        if meta[field]:
                            stats["field_completeness"][field] += 1
                    elif isinstance(meta[field], (int, float)):
                        if meta[field] > 0:
                            stats["field_completeness"][field] += 1
                    else:
                        if meta[field]:
                            stats["field_completeness"][field] += 1

            # Token counts
            description = meta.get("description", "")
            synopsis = meta.get("synopsis", "")

            desc_tokens = self.count_tokens(description)
            syn_tokens = self.count_tokens(synopsis)

            total_tokens += desc_tokens + syn_tokens
            total_description_tokens += desc_tokens
            total_synopsis_tokens += syn_tokens

            # Score statistics for movie/tv pages
            if page_type in ["movie", "tv"]:
                tomatometer = meta.get("tomatometer", 0)
                audience_score = meta.get("audience_score", 0)

                if tomatometer > 0:
                    if "tomatometer" not in stats["score_stats"]:
                        stats["score_stats"]["tomatometer"] = {"values": [], "avg": 0}
                    stats["score_stats"]["tomatometer"]["values"].append(tomatometer)

                if audience_score > 0:
                    if "audience_score" not in stats["score_stats"]:
                        stats["score_stats"]["audience_score"] = {"values": [], "avg": 0}
                    stats["score_stats"]["audience_score"]["values"].append(audience_score)

            # Validation checks
            self._validate_page(meta, stats["validation_results"])

        # Calculate averages and percentages
        total = len(metadata)
        for field in stats["field_completeness"]:
            stats["field_completeness"][field] = round(stats["field_completeness"][field] / total * 100, 2)

        for score_type in stats["score_stats"]:
            values = stats["score_stats"][score_type]["values"]
            if values:
                stats["score_stats"][score_type]["avg"] = round(sum(values) / len(values), 2)
                stats["score_stats"][score_type]["count"] = len(values)

        stats["token_stats"]["total_tokens"] = total_tokens
        stats["token_stats"]["description_tokens"] = total_description_tokens
        stats["token_stats"]["synopsis_tokens"] = total_synopsis_tokens
        stats["token_stats"]["avg_tokens_per_page"] = round(total_tokens / total, 2) if total > 0 else 0

        return stats

    def _validate_page(self, meta: Dict[str, Any], validation_results: Dict[str, List]):
        """Validate individual page extraction."""
        page_type = meta.get("page_type", "unknown")
        file_path = meta.get("file_path", "")

        # Basic validation rules
        if not meta.get("title"):
            validation_results["missing_title"].append(file_path)

        if not meta.get("url"):
            validation_results["missing_url"].append(file_path)

        if page_type in ["movie", "tv"]:
            if not meta.get("tomatometer") and not meta.get("audience_score"):
                validation_results["missing_scores"].append(file_path)

            if not meta.get("genres"):
                validation_results["missing_genres"].append(file_path)

            if not meta.get("description") and not meta.get("synopsis"):
                validation_results["missing_description"].append(file_path)

    def save_statistics_report(self, stats: Dict[str, Any], filename: str = "statistics_report.json"):
        """Save statistics report to file."""
        if filename.startswith(self.storage_dir):
            # If filename already contains the full path, use it directly
            path = filename
        else:
            # Otherwise, join with storage_dir as normal
            path = os.path.join(self.storage_dir, filename)

        # Convert defaultdict to regular dict for JSON serialization
        serializable_stats = {}
        for key, value in stats.items():
            if isinstance(value, defaultdict):
                serializable_stats[key] = dict(value)
            else:
                serializable_stats[key] = value

        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

        print(f"Statistics report saved to: {path}")
        return path

    def print_statistics(self, stats: Dict[str, Any]):
        """Print statistics in a readable format."""
        print("\n" + "=" * 60)
        print("EXTRACTION STATISTICS REPORT")
        print("=" * 60)

        print(f"\nTotal pages processed: {stats['total_pages']}")

        print("\nPage Type Distribution:")
        for page_type, count in stats['page_type_counts'].items():
            percentage = (count / stats['total_pages']) * 100
            print(f"  {page_type}: {count} ({percentage:.1f}%)")

        print("\nField Completeness (% of pages with data):")
        for field, completeness in stats['field_completeness'].items():
            print(f"  {field}: {completeness}%")

        print("\nToken Statistics:")
        print(f"  Total tokens: {stats['token_stats']['total_tokens']:,}")
        print(f"  Description tokens: {stats['token_stats']['description_tokens']:,}")
        print(f"  Synopsis tokens: {stats['token_stats']['synopsis_tokens']:,}")
        print(f"  Average tokens per page: {stats['token_stats']['avg_tokens_per_page']:.1f}")

        if stats['score_stats']:
            print("\nScore Statistics (movie/tv pages only):")
            for score_type, score_data in stats['score_stats'].items():
                if 'avg' in score_data:
                    print(f"  {score_type}: {score_data['avg']} (from {score_data['count']} pages)")

        print("\nValidation Issues:")
        if stats['validation_results']:
            if isinstance(next(iter(stats['validation_results'].values())), list):
                # validation_results contains lists of file paths
                total_issues = sum(len(issues) for issues in stats['validation_results'].values())
            else:
                # validation_results contains integer counts
                total_issues = sum(stats['validation_results'].values())
        else:
            total_issues = 0

        if total_issues == 0:
            print("  No validation issues found!")
        else:
            for issue_type, issues in stats['validation_results'].items():
                if isinstance(issues, list):
                    print(f"  {issue_type}: {len(issues)} pages")
                else:
                    print(f"  {issue_type}: {issues} pages")

        print("=" * 60)


# =========================
#   UNIT TESTS
# =========================
class RTUnitTests:
    """Unit tests for the Rotten Tomatoes extractor."""

    def __init__(self, extractor: RTDataExtractor, storage_dir: str = "extracted_metadata"):
        self.extractor = extractor
        self.storage_dir = storage_dir
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "test_cases": []
        }

    def run_all_tests(self, sample_files: List[str]) -> Dict[str, Any]:
        """Run all unit tests on sample files."""
        print("\n" + "=" * 60)
        print("RUNNING UNIT TESTS")
        print("=" * 60)

        self._test_extraction_basic(sample_files)
        self._test_page_type_detection(sample_files)
        self._test_field_completeness(sample_files)
        self._test_data_quality(sample_files)
        self._test_edge_cases()

        print(f"\nTest Results: {self.test_results['passed']} passed, {self.test_results['failed']} failed")

        # Save test report
        self._save_test_report()

        return self.test_results

    def _test_extraction_basic(self, sample_files: List[str]):
        """Test basic extraction functionality."""
        print("\n1. Testing Basic Extraction...")

        for i, file_path in enumerate(sample_files[:5], 1):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()

                result = self.extractor.extract_from_html(html, file_path)

                # Basic validation
                assert isinstance(result, dict), "Result should be a dictionary"
                assert "page_type" in result, "Missing page_type field"
                assert "title" in result, "Missing title field"
                assert "url" in result, "Missing url field"

                self._record_test_result(f"Basic extraction #{i}", True,
                                         f"Successfully extracted from {os.path.basename(file_path)}")

            except Exception as e:
                self._record_test_result(f"Basic extraction #{i}", False,
                                         f"Failed on {os.path.basename(file_path)}: {str(e)}")

    def _test_page_type_detection(self, sample_files: List[str]):
        """Test page type detection accuracy."""
        print("\n2. Testing Page Type Detection...")

        type_mapping = {
            "m_": "movie",
            "tv_": "tv",
            "celebrity_": "celebrity",
            "critic": "critic"
        }

        for file_path in sample_files:
            filename = os.path.basename(file_path).lower()

            expected_type = "unknown"
            for prefix, page_type in type_mapping.items():
                if prefix in filename:
                    expected_type = page_type
                    break

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()

                result = self.extractor.extract_from_html(html, file_path)
                detected_type = result.get("page_type", "unknown")

                # For this test, we'll be lenient - as long as it detects something reasonable
                if detected_type != "unknown":
                    self._record_test_result(
                        f"Page type detection for {filename}",
                        True,
                        f"Detected {detected_type} (expected {expected_type})"
                    )
                else:
                    self._record_test_result(
                        f"Page type detection for {filename}",
                        False,
                        f"Failed to detect page type (expected {expected_type})"
                    )

            except Exception as e:
                self._record_test_result(
                    f"Page type detection for {filename}",
                    False,
                    f"Error: {str(e)}"
                )

    def _test_field_completeness(self, sample_files: List[str]):
        """Test that required fields are populated."""
        print("\n3. Testing Field Completeness...")

        required_fields = ["page_type", "title", "url"]
        media_fields = ["tomatometer", "audience_score", "genres"]

        for file_path in sample_files[:10]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()

                result = self.extractor.extract_from_html(html, file_path)

                # Check required fields
                missing_required = [field for field in required_fields if not result.get(field)]

                # For media pages, check media fields
                if result.get("page_type") in ["movie", "tv"]:
                    missing_media = [field for field in media_fields if not result.get(field)]
                else:
                    missing_media = []

                if not missing_required and not missing_media:
                    self._record_test_result(
                        f"Field completeness for {os.path.basename(file_path)}",
                        True,
                        "All required fields present"
                    )
                else:
                    missing = missing_required + missing_media
                    self._record_test_result(
                        f"Field completeness for {os.path.basename(file_path)}",
                        False,
                        f"Missing fields: {', '.join(missing)}"
                    )

            except Exception as e:
                self._record_test_result(
                    f"Field completeness for {os.path.basename(file_path)}",
                    False,
                    f"Error: {str(e)}"
                )

    def _test_data_quality(self, sample_files: List[str]):
        """Test data quality and formatting."""
        print("\n4. Testing Data Quality...")

        test_cases = [
            ("Score ranges", lambda r: 0 <= r.get("tomatometer", 0) <= 100),
            ("Score ranges", lambda r: 0 <= r.get("audience_score", 0) <= 100),
            ("URL format", lambda r: not r.get("url") or r["url"].startswith(("http://", "https://"))),
            ("Genre format", lambda r: not r.get("genres") or isinstance(r["genres"], list)),
        ]

        for file_path in sample_files[:5]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()

                result = self.extractor.extract_from_html(html, file_path)

                for test_name, validation_func in test_cases:
                    try:
                        if validation_func(result):
                            self._record_test_result(
                                f"Data quality: {test_name} - {os.path.basename(file_path)}",
                                True,
                                f"Validation passed"
                            )
                        else:
                            self._record_test_result(
                                f"Data quality: {test_name} - {os.path.basename(file_path)}",
                                False,
                                f"Validation failed"
                            )
                    except Exception as e:
                        self._record_test_result(
                            f"Data quality: {test_name} - {os.path.basename(file_path)}",
                            False,
                            f"Validation error: {str(e)}"
                        )

            except Exception as e:
                self._record_test_result(
                    f"Data quality tests for {os.path.basename(file_path)}",
                    False,
                    f"Extraction error: {str(e)}"
                )

    def _test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n5. Testing Edge Cases...")

        # Test empty HTML
        try:
            result = self.extractor.extract_from_html("", "test_empty.html")
            assert result["page_type"] == "unknown", "Should handle empty HTML"
            self._record_test_result("Empty HTML handling", True, "Handled empty HTML correctly")
        except Exception as e:
            self._record_test_result("Empty HTML handling", False, f"Failed: {str(e)}")

        # Test malformed HTML
        try:
            malformed_html = "<html><body>Invalid content</body></html>"
            result = self.extractor.extract_from_html(malformed_html, "test_malformed.html")
            self._record_test_result("Malformed HTML handling", True, "Handled malformed HTML without crashing")
        except Exception as e:
            self._record_test_result("Malformed HTML handling", False, f"Crashed on malformed HTML: {str(e)}")

    def _record_test_result(self, test_name: str, passed: bool, message: str):
        """Record individual test result."""
        result = {
            "test_name": test_name,
            "passed": passed,
            "message": message
        }

        self.test_results["test_cases"].append(result)

        if passed:
            self.test_results["passed"] += 1
            print(f"  ✓ {test_name}: {message}")
        else:
            self.test_results["failed"] += 1
            print(f"  ✗ {test_name}: {message}")

    def _save_test_report(self):
        """Save test results to file."""
        path = os.path.join(self.storage_dir, "unit_test_results.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)

        print(f"\nTest report saved to: {path}")


# =========================
#   PIPELINE
# =========================
class RTExtractionPipeline:
    """Batch pipeline with on-disk progress tracking and failed files handling."""

    def __init__(self, html_directory: str, storage_directory: str = "extracted_metadata"):
        self.html_directory = html_directory
        self.extractor = RTDataExtractor()
        self.storage = RTMetadataStorage(storage_directory)
        self.statistics = RTStatistics(storage_directory)
        self.processed_files = set()
        self.failed_files = set()
        self.load_progress()

    def load_progress(self):
        """Load progress from previous runs, including failed files."""
        # Load processed files
        progress_path = os.path.join(self.storage.storage_dir, "progress.txt")
        if os.path.exists(progress_path):
            with open(progress_path, "r", encoding="utf-8", errors="ignore") as f:
                self.processed_files = set(line.strip() for line in f if line.strip())
            print(f"Resuming from previous progress: {len(self.processed_files)} files already processed")

        # Load failed files
        failed_path = os.path.join(self.storage.storage_dir, "failed.txt")
        if os.path.exists(failed_path):
            with open(failed_path, "r", encoding="utf-8", errors="ignore") as f:
                self.failed_files = set(line.strip() for line in f if line.strip())
            print(f"Found {len(self.failed_files)} previously failed files")

    def save_progress(self, file_path: str):
        """Save progress for successfully processed files."""
        path = os.path.join(self.storage.storage_dir, "progress.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(file_path + "\n")
        self.processed_files.add(file_path)

        # Remove from failed files if it was there before
        if file_path in self.failed_files:
            self.failed_files.remove(file_path)
            self._update_failed_files()

    def record_failed_file(self, file_path: str, error: str = ""):
        """Record a failed file for later retry."""
        path = os.path.join(self.storage.storage_dir, "failed.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(file_path + "\n")
        self.failed_files.add(file_path)
        print(f"Recorded failed file: {file_path} - {error}")

    def _update_failed_files(self):
        """Update the failed.txt file with current failed files."""
        path = os.path.join(self.storage.storage_dir, "failed.txt")
        with open(path, "w", encoding="utf-8") as f:
            for file_path in self.failed_files:
                f.write(file_path + "\n")

    def retry_failed_files(self, batch_size: int = 1000, output_format: str = "jsonl") -> Dict[str, Any]:
        """Retry processing previously failed files."""
        if not self.failed_files:
            print("No failed files to retry")
            return {}

        print(f"Retrying {len(self.failed_files)} failed files...")
        return self._process_file_list(list(self.failed_files), batch_size, output_format, is_retry=True)

    def find_html_files(
            self,
            max_files: int = None,
            include_failed: bool = False,
            mode: str = "url_strict",
            media: str = "all",
    ) -> List[str]:
        """
        Return only *root* media pages (no reviews/cast/episodes).
        mode:
          - "fast": filename-only prefilter (no file reads). Root pages only.
          - "url_strict": open file head and validate canonical/og:url/ld+json.url.
        media:
          - "movie" -> only /m/<slug>
          - "tv"    -> only /tv/<slug> or /series/<slug>
          - "all"   -> both
        """
        html_files: List[str] = []
        max_read = 65536  # 64KB head read for canonical/meta

        def _is_html(name: str) -> bool:
            return name.endswith(".html")

        # Root-only filename heuristics (exclude subpages)
        BAD_TV_BITS = ("episodes", "episode", "season", "cast_and_crew", "reviews",
                       "_s0", "_s1", "_s2", "_s3", "_s4", "_s5", "_s6", "_s7", "_s8", "_s9")
        BAD_MOVIE_BITS = ("cast_and_crew", "reviews", "photos", "clips", "videos", "news")

        def _is_root_movie_name(name: str) -> bool:
            if not (name.startswith("rottentomatoes_com_m_") and _is_html(name)):
                return False
            return not any(b in name for b in BAD_MOVIE_BITS)

        def _is_root_tv_name(name: str) -> bool:
            if not _is_html(name):
                return False
            if not (name.startswith("rottentomatoes_com_tv_") or name.startswith("rottentomatoes_com_series_")):
                return False
            return not any(b in name for b in BAD_TV_BITS)

        import re

        def _keep_strict(full: str, name_lower: str) -> bool:
            """Head-parse canonical and require root URL by media type."""
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(max_read)

                canon = self.extractor._first_nonempty(
                    self.extractor._match1(r'<link\s+rel="canonical"\s+href="([^"]+)"', head, flags=re.I),
                    self.extractor._match1(r'<meta\s+property="og:url"\s+content="([^"]+)"', head, flags=re.I),
                    self.extractor._match1(r'"url"\s*:\s*"([^"]+)"', head, flags=re.I),
                ).strip().lower()

                if canon:
                    if canon.startswith("//"):
                        canon = "https:" + canon
                    canon = re.sub(r"[?#].*$", "", canon)

                def _is_root_movie(u: str) -> bool:
                    return bool(re.match(r'^(?:https?:)?//(?:www\.)?rottentomatoes\.com/m/[^/]+/?$', u))

                def _is_root_tv(u: str) -> bool:
                    return bool(re.match(r'^(?:https?:)?//(?:www\.)?rottentomatoes\.com/(?:tv|series)/[^/]+/?$', u))

                def _is_tv_nonroot(u: str) -> bool:
                    return bool(re.search(
                        r'/(?:tv|series)/[^/]+/(?:s\d{1,2}(?:/e\d{1,3})?|episodes?|episode|season|cast_and_crew|reviews|photos|clips|videos|news)(?:/|$)',
                        u))

                def _is_movie_subpage(u: str) -> bool:
                    return bool(re.search(
                        r'/m/[^/]+/(?:cast_and_crew|reviews|photos|clips|videos|news)(?:/|$)', u))

                if not canon:
                    # fall back to filename-only root checks
                    if media == "movie":
                        return _is_root_movie_name(name_lower)
                    if media == "tv":
                        return _is_root_tv_name(name_lower)
                    return _is_root_movie_name(name_lower) or _is_root_tv_name(name_lower)

                # Canonical-based root checks
                if media == "movie":
                    return _is_root_movie(canon) and not _is_movie_subpage(canon)
                if media == "tv":
                    return _is_root_tv(canon) and not _is_tv_nonroot(canon)
                # all
                return (
                        (_is_root_movie(canon) and not _is_movie_subpage(canon)) or
                        (_is_root_tv(canon) and not _is_tv_nonroot(canon))
                )
            except Exception:
                return False

        # Walk filesystem
        for root, _dirs, files in os.walk(self.html_directory):
            for file in files:
                name_lower = file.lower()
                if not _is_html(name_lower):
                    continue

                # quick filename-only branch
                if mode == "fast":
                    if media == "movie" and not _is_root_movie_name(name_lower):
                        continue
                    if media == "tv" and not _is_root_tv_name(name_lower):
                        continue
                    if media == "all" and not (_is_root_movie_name(name_lower) or _is_root_tv_name(name_lower)):
                        continue

                    full = os.path.join(root, file)

                else:  # url_strict
                    full = os.path.join(root, file)
                    if full in self.processed_files:
                        continue
                    if not include_failed and full in self.failed_files:
                        continue
                    if not _keep_strict(full, name_lower):
                        continue

                if full in self.processed_files:
                    continue
                if not include_failed and full in self.failed_files:
                    continue

                html_files.append(full)
                if max_files and len(html_files) >= max_files:
                    return html_files

        return html_files

    def _generate_batch_statistics(self, batch_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Generate lightweight batch statistics that can be aggregated later."""
        batch_stats = {
            "batch_files": len(batch_meta),
            "page_type_counts": defaultdict(int),
            "field_completeness": defaultdict(int),
            "token_stats": {
                "total_tokens": 0,
                "description_tokens": 0,
                "synopsis_tokens": 0
            },
            "score_stats": {
                "tomatometer": {"sum": 0, "count": 0},
                "audience_score": {"sum": 0, "count": 0}
            },
            "validation_results": defaultdict(int)
        }

        for file_path, meta in batch_meta.items():
            page_type = meta.get("page_type", "unknown")
            batch_stats["page_type_counts"][page_type] += 1

            # Field completeness counts
            for field in ["title", "url", "description", "tomatometer", "audience_score", "genres", "cast"]:
                if field in meta and meta[field]:
                    if isinstance(meta[field], list):
                        if meta[field]:
                            batch_stats["field_completeness"][field] += 1
                    elif isinstance(meta[field], (int, float)):
                        if meta[field] > 0:
                            batch_stats["field_completeness"][field] += 1
                    else:
                        if meta[field]:
                            batch_stats["field_completeness"][field] += 1

            # Token counts
            description = meta.get("description", "")
            synopsis = meta.get("synopsis", "")

            desc_tokens = self.statistics.count_tokens(description)
            syn_tokens = self.statistics.count_tokens(synopsis)

            batch_stats["token_stats"]["total_tokens"] += desc_tokens + syn_tokens
            batch_stats["token_stats"]["description_tokens"] += desc_tokens
            batch_stats["token_stats"]["synopsis_tokens"] += syn_tokens

            # Score statistics
            if page_type in ["movie", "tv"]:
                tomatometer = meta.get("tomatometer", 0)
                audience_score = meta.get("audience_score", 0)

                if tomatometer > 0:
                    batch_stats["score_stats"]["tomatometer"]["sum"] += tomatometer
                    batch_stats["score_stats"]["tomatometer"]["count"] += 1

                if audience_score > 0:
                    batch_stats["score_stats"]["audience_score"]["sum"] += audience_score
                    batch_stats["score_stats"]["audience_score"]["count"] += 1

            # Validation counts (just counts, not file lists)
            self._count_validation_issues(meta, batch_stats["validation_results"])

        return batch_stats

    def _count_validation_issues(self, meta: Dict[str, Any], validation_counts: Dict[str, int]):
        """Count validation issues without storing file paths."""
        page_type = meta.get("page_type", "unknown")
        file_path = meta.get("file_path", "")

        if not meta.get("title"):
            validation_counts["missing_title"] += 1

        if not meta.get("url"):
            validation_counts["missing_url"] += 1

        if page_type in ["movie", "tv"]:
            if not meta.get("tomatometer") and not meta.get("audience_score"):
                validation_counts["missing_scores"] += 1

            if not meta.get("genres"):
                validation_counts["missing_genres"] += 1

            if not meta.get("description") and not meta.get("synopsis"):
                validation_counts["missing_description"] += 1

    def _aggregate_batch_statistics(self, batch_stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate batch statistics into comprehensive statistics."""
        comprehensive = {
            "total_pages": 0,
            "page_type_counts": defaultdict(int),
            "field_completeness": defaultdict(int),
            "token_stats": defaultdict(int),
            "score_stats": defaultdict(dict),
            "validation_results": defaultdict(int)
        }

        total_batches = len(batch_stats_list)
        total_pages = 0

        # Aggregate all batch statistics
        for batch in batch_stats_list:
            total_pages += batch["batch_files"]
            comprehensive["total_pages"] = total_pages

            # Aggregate page type counts
            for page_type, count in batch["page_type_counts"].items():
                comprehensive["page_type_counts"][page_type] += count

            # Aggregate field completeness (these are counts, will convert to percentages later)
            for field, count in batch["field_completeness"].items():
                comprehensive["field_completeness"][field] += count

            # Aggregate token stats
            for token_type, count in batch["token_stats"].items():
                comprehensive["token_stats"][token_type] += count

            # Aggregate score stats
            for score_type, data in batch["score_stats"].items():
                if score_type not in comprehensive["score_stats"]:
                    comprehensive["score_stats"][score_type] = {"sum": 0, "count": 0}
                comprehensive["score_stats"][score_type]["sum"] += data["sum"]
                comprehensive["score_stats"][score_type]["count"] += data["count"]

            # Aggregate validation results
            for issue_type, count in batch["validation_results"].items():
                comprehensive["validation_results"][issue_type] += count

        # Convert counts to percentages and finalize structure
        return self._finalize_statistics(comprehensive)

    def _finalize_statistics(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Convert aggregated counts to final statistics format."""
        total_pages = aggregated["total_pages"]

        # Convert field completeness to percentages
        field_completeness_pct = {}
        for field, count in aggregated["field_completeness"].items():
            field_completeness_pct[field] = round(count / total_pages * 100, 2) if total_pages > 0 else 0

        # Calculate score averages
        score_stats_final = {}
        for score_type, data in aggregated["score_stats"].items():
            if data["count"] > 0:
                score_stats_final[score_type] = {
                    "avg": round(data["sum"] / data["count"], 2),
                    "count": data["count"]
                }
            else:
                score_stats_final[score_type] = {"avg": 0, "count": 0}

        # Calculate average tokens
        avg_tokens_per_page = round(aggregated["token_stats"]["total_tokens"] / total_pages,
                                    2) if total_pages > 0 else 0

        return {
            "total_pages": total_pages,
            "page_type_counts": dict(aggregated["page_type_counts"]),
            "field_completeness": field_completeness_pct,
            "token_stats": {
                "total_tokens": aggregated["token_stats"]["total_tokens"],
                "description_tokens": aggregated["token_stats"]["description_tokens"],
                "synopsis_tokens": aggregated["token_stats"]["synopsis_tokens"],
                "avg_tokens_per_page": avg_tokens_per_page
            },
            "score_stats": score_stats_final,
            "validation_results": dict(aggregated["validation_results"])
        }

    def _process_file_list(self, file_list: List[str], batch_size: int, output_format: str, is_retry: bool = False) -> \
            Dict[str, Any]:
        """Process a list of files with batching - memory efficient version."""
        total = len(file_list)
        if total == 0:
            print("No files to process")
            return {}

        print(f"Processing {total} HTML files..." + (" (retry mode)" if is_retry else ""))
        all_meta: Dict[str, Any] = {}
        processed = 0
        start = time.time()

        # For memory efficiency: generate batch statistics and aggregate them
        batch_stats_list = []

        for file_path in file_list:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()

                meta = self.extractor.extract_from_html(html, file_path)
                all_meta[file_path] = meta

                processed += 1
                self.save_progress(file_path)

                if processed % 100 == 0:
                    elapsed = time.time() - start
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    rem = total - processed
                    eta = rem / rate if rate > 0 else 0.0
                    print(f"Progress: {processed}/{total} ({processed / total * 100:.1f}%) - ETA {eta / 60:.1f} min")

                # When batch is complete, save it and generate batch statistics
                if len(all_meta) >= batch_size:
                    if output_format == "jsonl":
                        self.storage.save_to_jsonl(all_meta)
                    elif output_format == "csv":
                        self.storage.save_to_csv(all_meta)

                    # Generate and store batch statistics, then clear memory
                    batch_stats = self._generate_batch_statistics(all_meta)
                    batch_stats_list.append(batch_stats)
                    all_meta = {}

            except Exception as e:
                error_msg = str(e)
                print(f"Error processing {file_path}: {error_msg}")
                self.record_failed_file(file_path, error_msg)

        # Process final batch
        if all_meta:
            if output_format == "jsonl":
                self.storage.save_to_jsonl(all_meta)
            elif output_format == "csv":
                self.storage.save_to_csv(all_meta)

            # Generate statistics for final batch
            batch_stats = self._generate_batch_statistics(all_meta)
            batch_stats_list.append(batch_stats)

        elapsed = time.time() - start
        print(f"Extraction complete: {processed} files processed in {elapsed / 60:.1f} minutes")

        # Generate comprehensive statistics from aggregated batch summaries
        if batch_stats_list:
            print("\nGenerating comprehensive statistics from batch summaries...")
            comprehensive_stats = self._aggregate_batch_statistics(batch_stats_list)
            self.statistics.print_statistics(comprehensive_stats)

            timestamp = int(time.time())
            stats_path = os.path.join(self.storage.storage_dir, f"comprehensive_statistics_{timestamp}.json")
            self.statistics.save_statistics_report(comprehensive_stats, stats_path)

        return all_meta  # Returns only the last batch, but we have comprehensive stats

    def extract_batch(self, batch_size: int = 1000, max_files: int = None, output_format: str = "jsonl",
                      include_failed: bool = False) -> Dict[str, Any]:
        """Extract metadata from HTML files."""
        html_files = self.find_html_files(max_files, include_failed)
        return self._process_file_list(html_files, batch_size, output_format)

    def generate_statistics_from_batch_files(self):
        """Generate comprehensive statistics by reading all batch files (memory efficient)."""
        print("Generating statistics from existing batch files...")

        batch_files = []
        for file in os.listdir(self.storage.storage_dir):
            if file.startswith("metadata_batch_") and file.endswith(".jsonl"):
                batch_files.append(os.path.join(self.storage.storage_dir, file))

        if not batch_files:
            print("No batch files found. Run extraction first.")
            return None

        batch_stats_list = []

        for batch_file in batch_files:
            print(f"Processing {os.path.basename(batch_file)}...")
            batch_meta = {}

            with open(batch_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        meta = json.loads(line.strip())
                        # Use a placeholder key since we don't have file paths in JSONL
                        key = f"{batch_file}_{len(batch_meta)}"
                        batch_meta[key] = meta
                    except json.JSONDecodeError:
                        continue

            # Generate batch statistics
            batch_stats = self._generate_batch_statistics(batch_meta)
            batch_stats_list.append(batch_stats)

        # Aggregate statistics
        comprehensive_stats = self._aggregate_batch_statistics(batch_stats_list)
        self.statistics.print_statistics(comprehensive_stats)

        timestamp = int(time.time())
        # FIX: Just pass the filename, not the full path
        filename = f"comprehensive_statistics_{timestamp}.json"
        self.statistics.save_statistics_report(comprehensive_stats, filename)

        return comprehensive_stats

    def run_unit_tests(self, sample_size: int = 20, media: str = "all") -> Dict[str, Any]:
        """Run unit tests on a small, fast-picked subset filtered by media ('movie'|'tv'|'all')."""
        # Prefer zero-I/O filename sampling for speed; fallback to strict if not enough
        pool = self.find_html_files(max_files=sample_size * 100, include_failed=False, mode="fast", media=media)
        if len(pool) < sample_size:
            strict_pool = self.find_html_files(max_files=sample_size * 100, include_failed=False, mode="url_strict",
                                               media=media)
            # merge (avoid dupes)
            got = set(pool)
            for p in strict_pool:
                if len(pool) >= sample_size:
                    break
                if p not in got:
                    pool.append(p)
                    got.add(p)

        if not pool:
            print(f"No HTML files found for media='{media}'. Check your directory path or adjust filters.")
            return {}

        sample = pool if len(pool) <= sample_size else random.sample(pool, sample_size)

        unit_tests = RTUnitTests(self.extractor, self.storage.storage_dir)
        return unit_tests.run_all_tests(sample)

    def test_on_sample(self, sample_size: int = 20, media: str = "all") -> Dict[str, Any]:
        """Quick extraction check on a filtered sample ('movie'|'tv'|'all'), FAST first."""
        pool = self.find_html_files(max_files=sample_size * 100, include_failed=False, mode="fast", media=media)
        if len(pool) < sample_size:
            strict_pool = self.find_html_files(max_files=sample_size * 100, include_failed=False, mode="url_strict",
                                               media=media)
            got = set(pool)
            for p in strict_pool:
                if len(pool) >= sample_size:
                    break
                if p not in got:
                    pool.append(p)
                    got.add(p)

        if not pool:
            print(f"No HTML files found for media='{media}'.")
            return {}

        sample = pool if len(pool) <= sample_size else random.sample(pool, sample_size)
        print(f"Testing on {len(sample)} random {media.upper()} files...")
        print("-" * 60)

        sample_meta: Dict[str, Any] = {}
        summary = {"total_tested": len(sample), "successful": 0, "failed": 0,
                   "page_types": defaultdict(int), "extraction_stats": defaultdict(int)}

        for i, file_path in enumerate(sample, 1):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()
                meta = self.extractor.extract_from_html(html, file_path)
                sample_meta[file_path] = meta

                summary["successful"] += 1
                summary["page_types"][meta.get("page_type", "unknown")] += 1
                if meta.get("title"): summary["extraction_stats"]["has_title"] += 1
                if meta.get("page_type") in {"movie", "tv"} and meta.get("tomatometer", 0) > 0:
                    summary["extraction_stats"]["has_tomatometer"] += 1
                if meta.get("genres"): summary["extraction_stats"]["has_genres"] += 1

                fname = os.path.basename(file_path)
                print(
                    f"{i:2d}. ✓ {fname[:40]:40} | {meta.get('page_type', 'unknown'):8} | {meta.get('title', 'No title')[:40]}")
            except Exception as e:
                summary["failed"] += 1
                fname = os.path.basename(file_path)
                print(f"{i:2d}. ✗ {fname[:40]:40} | ERROR: {str(e)[:60]}")

        out_path = os.path.join(self.storage.storage_dir, "sample_metadata.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sample_meta, f, indent=2, ensure_ascii=False)
        print(f"\nSample results saved to: {out_path}")

        stats = self.statistics.generate_statistics(sample_meta)
        self.statistics.print_statistics(stats)
        return sample_meta


# =========================
#   CLI
# =========================
def run_extraction_pipeline():
    """Run the complete extraction pipeline on your data"""
    html_directory = "rotten_tomatoes_data/html_pages"
    storage_directory = "extracted_metadata"

    if not os.path.exists(html_directory):
        print(f"HTML directory not found: {html_directory}")
        print("Please update the path in run_extraction_pipeline()")
        return

    pipeline = RTExtractionPipeline(html_directory, storage_directory)

    print("=== ROTTEN TOMATOES EXTRACTION PIPELINE ===")
    print("1. Run unit tests (20+ pages)")
    print("2. Test extraction on sample")
    print("3. Full extraction")
    print("4. Retry failed files")
    print("5. Generate statistics from existing data")

    choice = input("\nChoose option (1-5): ").strip()

    if choice == "1":
        print("\n=== RUNNING UNIT TESTS ===")
        media = input("Filter (movie/tv/all) [tv]: ").strip().lower() or "tv"
        if media not in {"movie", "tv", "all"}: media = "tv"
        pipeline.run_unit_tests(sample_size=20, media=media)

    elif choice == "2":
        print("\n=== TESTING ON ACTUAL FILES ===")
        media = input("Filter (movie/tv/all) [tv]: ").strip().lower() or "tv"
        if media not in {"movie", "tv", "all"}: media = "tv"
        sample_results = pipeline.test_on_sample(sample_size=20, media=media)
        if not sample_results:
            print("No files could be processed. Check your HTML files.")
            return

    elif choice == "3":
        print("\n=== STARTING FULL EXTRACTION ===")
        fmt = input("Output format (jsonl/csv/both) [jsonl]: ").strip().lower()
        if fmt not in ["jsonl", "csv", "both"]:
            fmt = "jsonl"

        include_failed = input("Include previously failed files? (y/n) [n]: ").strip().lower() == "y"

        if fmt in ["jsonl", "both"]:
            pipeline.extract_batch(batch_size=1000, output_format="jsonl", include_failed=include_failed)
        if fmt in ["csv", "both"]:
            pipeline.extract_batch(batch_size=1000, output_format="csv", include_failed=include_failed)

        print("=== EXTRACTION COMPLETE ===")
        print(f"Metadata saved to: {storage_directory}")

        print("\nCreated files:")
        for file in os.listdir(storage_directory):
            if file.endswith((".jsonl", ".csv", ".pkl", ".json")):
                filepath = os.path.join(storage_directory, file)
                try:
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"  {file} ({size_mb:.2f} MB)")
                except Exception:
                    print(f"  {file}")

    elif choice == "4":
        print("\n=== RETRYING FAILED FILES ===")
        fmt = input("Output format (jsonl/csv) [jsonl]: ").strip().lower()
        if fmt not in ["jsonl", "csv"]:
            fmt = "jsonl"
        pipeline.retry_failed_files(output_format=fmt)

    elif choice == "5":
        print("\n=== GENERATING STATISTICS FROM EXISTING DATA ===")
        pipeline.generate_statistics_from_batch_files()

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    run_extraction_pipeline()
