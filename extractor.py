import bz2
import glob
import json
import os
import re
import sys
import time
from io import TextIOWrapper
from typing import Any, Dict, List, Optional, Iterator
from html import unescape

# ----- Spark -----
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

# ----- Optional downloader helper -----
try:
    import requests
except Exception:
    requests = None  # Downloader option will explain how to install requests if missing

os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ.setdefault("hadoop.home.dir", r"C:\hadoop")
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

print("Driver sys.executable:", sys.executable)
print("PYSPARK_PYTHON:", os.environ.get("PYSPARK_PYTHON"))

WIKI_SCHEMA = StructType([
    StructField("wiki_page_id", StringType(), True),
    StructField("wiki_title", StringType(), True),
    StructField("wiki_title_norm", StringType(), True),
    StructField("wiki_year", IntegerType(), True),
    StructField("wiki_directors", ArrayType(StringType()), True),
    StructField("wiki_starring", ArrayType(StringType()), True),
    StructField("wiki_languages", ArrayType(StringType()), True),
    StructField("wiki_countries", ArrayType(StringType()), True),
    StructField("wiki_released_raw", StringType(), True),
    StructField("wiki_runtime_minutes", IntegerType(), True),
    StructField("wiki_budget", StringType(), True),
    StructField("wiki_box_office", StringType(), True),
])


def _as_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _as_str(x):
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _as_list_str(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if x is not None and str(x).strip()]
    return [str(v).strip()] if str(v).strip() else []


def coerce_wiki(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "wiki_page_id": _as_str(rec.get("wiki_page_id")),
        "wiki_title": _as_str(rec.get("wiki_title")),
        "wiki_title_norm": _as_str(rec.get("wiki_title_norm")),
        "wiki_year": rec.get("wiki_year") if isinstance(rec.get("wiki_year"), int) else _as_int(rec.get("wiki_year")),
        "wiki_directors": _as_list_str(rec.get("wiki_directors")),
        "wiki_starring": _as_list_str(rec.get("wiki_starring")),
        "wiki_languages": _as_list_str(rec.get("wiki_languages")),
        "wiki_countries": _as_list_str(rec.get("wiki_countries")),
        "wiki_released_raw": _as_str(rec.get("wiki_released_raw")),
        "wiki_runtime_minutes": rec.get("wiki_runtime_minutes") if isinstance(rec.get("wiki_runtime_minutes"),
                                                                              int) else _as_int(
            rec.get("wiki_runtime_minutes")),
        "wiki_budget": _as_str(rec.get("wiki_budget")),
        "wiki_box_office": _as_str(rec.get("wiki_box_office")),
    }


# ============================================================
#  CONFIG (change from the menu)
# ============================================================
HTML_DIR = "rt_roots_html"
WIKI_DIR = r"D:\ProgrammingFun\VINFData\wiki_dumps\enwiki_latest"
OUT_DIR = r"D:\ProgrammingFun\VINFData\spark_out"
RT_PARQUET = os.path.join(OUT_DIR, "rt.parquet")
WIKI_PARQUET = os.path.join(OUT_DIR, "wiki.parquet")
JOIN_PARQUET = os.path.join(OUT_DIR, "rt_wiki_join.parquet")
JOIN_SAMPLE_DIR = os.path.join(OUT_DIR, "rt_wiki_join_sample_csv")
WIKI_LATEST_URL = "https://dumps.wikimedia.org/enwiki/latest/"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(WIKI_DIR, exist_ok=True)


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


# ============================================================
#  Normalizers & year helpers
# ============================================================
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def normalize_title(s: Optional[str]) -> str:
    if not s:
        return ""
    x = s.strip().lower()

    # 1) unify "&" and "and"
    x = x.replace("&", " and ")

    # 2) strip accents (so 'amélie' -> 'amelie')
    import unicodedata
    x = "".join(
        c for c in unicodedata.normalize("NFD", x)
        if unicodedata.category(c) != "Mn"
    )

    # 3) drop film suffixes and trailing parentheses
    x = re.sub(r"\s*\((?:\d{4}\s*)?film\)\s*$", "", x)
    x = re.sub(r"\s*\(film\)\s*$", "", x)
    x = re.sub(r"\s*\([^)]*\)\s*$", "", x)

    # 4) simple roman numerals → digits for common sequels
    roman_map = {
        " ii ": " 2 ",
        " iii ": " 3 ",
        " iv ": " 4 ",
        " v ": " 5 ",
        " vi ": " 6 ",
        " vii ": " 7 ",
        " viii ": " 8 ",
        " ix ": " 9 ",
    }
    for k, v in roman_map.items():
        x = x.replace(k, v)

    # 5) non-alnum → space, kill leading articles
    x = re.sub(r"[^a-z0-9]+", " ", x).strip()
    x = re.sub(r"^(the|a|an)\s+", "", x)
    return x


def year_from_text(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = YEAR_RE.search(s)
    return int(m.group(0)) if m else None


def year_from_rt_release(release_date: Optional[str]) -> Optional[int]:
    return year_from_text(release_date)


# ============================================================
#  Wikipedia parsing via REGEX (Infobox film)
# ============================================================
TITLE_OPEN = re.compile(r"<title>(.*?)</title>")
ID_OPEN = re.compile(r"<id>(\d+)</id>")
TEXT_OPEN = re.compile(r'<text[^>]*>(.*)', re.DOTALL)
TEXT_CLOSE = re.compile(r"(.*)</text>", re.DOTALL)
REDIRECT_RE = re.compile(r"#REDIRECT\s+\[\[([^\]]+)\]\]", re.IGNORECASE)

INFOBOX_FILM_RE = re.compile(r"\{\{\s*Infobox\s+film(.*?)}\}", re.IGNORECASE | re.DOTALL)
WIKI_REF_RE = re.compile(r"<ref.*?</ref>|<ref.*?/>", re.DOTALL | re.IGNORECASE)
WIKI_LINK_RE = re.compile(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]")
WIKI_TEMPLATE_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")

FIELD_TAIL = r"(?=\n\|\s*[A-Za-z_]+\s*=|\n\Z)"

KEY_PATTERNS = {
    "director": re.compile(
        rf"^\s*\|\s*director[s]?\s*=\s*(.+?){FIELD_TAIL}",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    ),
    "starring": re.compile(
        rf"^\s*\|\s*starring\s*=\s*(.+?){FIELD_TAIL}",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    ),
    "released": re.compile(
        rf"^\s*\|\s*released\s*=\s*(.+?){FIELD_TAIL}",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    ),
    "runtime": re.compile(
        rf"^\s*\|\s*(?:runtime|running\s*time)\s*=\s*(.+?){FIELD_TAIL}",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    ),
    "language": re.compile(
        rf"^\s*\|\s*language[s]?\s*=\s*(.+?){FIELD_TAIL}",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    ),
    "country": re.compile(
        rf"^\s*\|\s*country\s*=\s*(.+?){FIELD_TAIL}",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    ),
    "budget": re.compile(
        rf"^\s*\|\s*budget\s*=\s*(.+?){FIELD_TAIL}",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    ),
    "box_office": re.compile(
        rf"^\s*\|\s*box\s*office\s*=\s*(.+?){FIELD_TAIL}",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    ),
}


TEMPLATE_LIST_KEEP = re.compile(
    r"\{\{\s*(?:plainlist|hlist|ubl|unbulleted\s+list|nowrap)\s*\|([^{}]+)\}\}",
    re.IGNORECASE,
)
TEMPLATE_FILM_DATE = re.compile(r"\{\{\s*film\s*date\s*\|([^{}]+)\}\}", re.IGNORECASE)
TEMPLATE_START_DATE = re.compile(r"\{\{\s*start\s*date[^|}]*\|([^{}]+)\}\}", re.IGNORECASE)
TEMPLATE_RUNTIME = re.compile(r"\{\{\s*(?:runtime|running\s*time)\s*\|([^{}]+)\}\}", re.IGNORECASE)

_SIMPLE_LIST_TPL = re.compile(
    r"\{\{\s*(?:ubl|unbulleted list|plainlist|hlist|flatlist)\s*\|([^{}]*)\}\}",
    re.IGNORECASE
)
_LAST_ARG_TPL = re.compile(
    r"\{\{\s*(?:lang|nowrap|ill|nobold)\s*\|(?:[^{}]*\|)?([^{}|]+)\}\}",
    re.IGNORECASE
)
_SIMPLE_ANY_TPL = re.compile(
    r"\{\{\s*([^{|}]+)\s*\|([^{}]*)\}\}",
    re.IGNORECASE
)

def find_infobox_film_block(text: str) -> Optional[str]:
    s = text.lower().find("{{infobox film")
    if s == -1:
        return None
    depth = 0
    i = s
    n = len(text)
    while i < n:
        if i + 1 < n and text[i] == "{" and text[i+1] == "{":
            depth += 1
            i += 2
            continue
        if i + 1 < n and text[i] == "}" and text[i+1] == "}":
            depth -= 1
            i += 2
            if depth == 0:
                return text[s:i]
            continue
        i += 1
    return None

def _preserve_template_text(s: str) -> str:
    # 1) list-like templates → keep inner text
    s = TEMPLATE_LIST_KEEP.sub(r"\1", s)
    # 2) runtime templates → keep inner (e.g., '138|minutes' → '138 minutes')
    def _runtime_to_text(m):
        inner = m.group(1).replace("|", " ")
        return inner
    s = TEMPLATE_RUNTIME.sub(_runtime_to_text, s)
    # 3) date templates → keep only years (works for year extraction and display)
    def _keep_years(m):
        inner = m.group(1)
        yrs = YEAR_RE.findall(inner)  # full 4-digit years
        return " ".join(dict.fromkeys(yrs))  # de-dupe, keep order

    s = TEMPLATE_FILM_DATE.sub(_keep_years, s)
    s = TEMPLATE_START_DATE.sub(_keep_years, s)
    # 4) any remaining simple {{...}} → drop braces but keep inner text
    s = re.sub(r"\{\{([^{}]+)\}\}", r"\1", s)
    return s



def _clean_wiki_text(s: str) -> str:
    if not s:
        return ""
    s = _preserve_template_text(s)

    # Decode HTML entities: &lt;br&gt; → <br>
    s = unescape(s)

    # <br> variants → newline (useful for lists)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)

    # Remove <ref>...</ref>
    s = WIKI_REF_RE.sub(" ", s)

    # [[A|B]] / [[B]] → "B"
    s = WIKI_LINK_RE.sub(r"\1", s)

    # Drop leftover HTML tags
    s = HTML_TAG_RE.sub(" ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _split_listish(s: str) -> List[str]:
    # Global clean: templates, refs, [[links]], &lt;br&gt; etc.
    s = _clean_wiki_text(s)

    # Kill noisy "Plain list" / "Plainlist" text that leaks from templates
    s = re.sub(r"(?i)\bplain\s*list,?\s*", "", s)
    s = re.sub(r"(?i)\bplainlist,?\s*", "", s)

    # Turn bullet stars into separators: "* A * B * C" → "A|B|C"
    if "*" in s:
        s = re.sub(r"\s*\*\s*", "|", s)

    # Now split on all reasonable separators
    raw_parts = re.split(r"[\n,;•\|\u2022]+", s)

    out, seen = [], set()
    for p in raw_parts:
        p = p.strip()
        if not p:
            continue

        # Handle "narrator = Robert Evans", "country = Italy", "website = ...", etc.
        if "=" in p:
            # keep only the part after "="
            p = p.split("=", 1)[1].strip()
            if not p:
                continue

        if p and p not in seen:
            out.append(p)
            seen.add(p)

    return out


def extract_infobox_attrs(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    block = find_infobox_film_block(text)
    if not block:
        return out

    # Search keys inside the full balanced block
    for key, pat in KEY_PATTERNS.items():
        km = pat.search(block)
        if not km:
            continue
        raw = _clean_wiki_text(km.group(1))
        if key in ("director", "starring", "language", "country"):
            out[key] = _split_listish(raw)
        else:
            out[key] = raw

    # Year from 'released' or whole page as fallback
    yy = None
    for src in [out.get("released", ""), text]:
        if not yy:
            yy = year_from_text(src)
    if yy:
        out["year"] = int(yy)

    # Runtime minutes (keep regex as-is)
    if "runtime" in out:
        mm = re.search(r"(\d+)\s*min", out["runtime"], re.IGNORECASE)
        if mm:
            out["runtime_minutes"] = int(mm.group(1))
    return out

def parse_wiki_pages_partition(lines: Iterator[str]) -> Iterator[Dict[str, Any]]:
    buf = []
    inside = False
    for line in lines:
        if "<page>" in line:
            inside = True
            buf = [line]
            continue
        if inside:
            buf.append(line)
            if "</page>" in line:
                page = "\n".join(buf)
                inside = False
                tm = TITLE_OPEN.search(page)
                if not tm:
                    continue
                title = tm.group(1)
                if ":" in title:  # skip non-mainspace
                    continue
                im = ID_OPEN.search(page)
                page_id = im.group(1) if im else None
                text = ""
                txm = TEXT_OPEN.search(page)
                if txm:
                    t = txm.group(1)
                    cxm = TEXT_CLOSE.search(t)
                    text = (cxm.group(1) if cxm else t)
                if re.match(r"#REDIRECT\s+\[\[", text, re.IGNORECASE):
                    continue
                attrs = extract_infobox_attrs(text)
                if not attrs:
                    continue
                yield {
                    "wiki_page_id": page_id,
                    "wiki_title": title,
                    "wiki_title_norm": normalize_title(title),
                    "wiki_year": attrs.get("year"),
                    "wiki_directors": attrs.get("director", []),
                    "wiki_starring": attrs.get("starring", []),
                    "wiki_languages": attrs.get("language", []),
                    "wiki_countries": attrs.get("country", []),
                    "wiki_released_raw": attrs.get("released"),
                    "wiki_runtime_minutes": attrs.get("runtime_minutes"),
                    "wiki_budget": attrs.get("budget"),
                    "wiki_box_office": attrs.get("box_office"),
                }
                buf = []

def parse_wiki_redirects_partition(lines: Iterator[str]) -> Iterator[Dict[str, Any]]:
    """
    Parse pages that are *only* redirects, e.g.
    #REDIRECT [[Léon: The Professional]]
    We keep (redirect_title_norm -> target_title_norm) so we can later
    map RT titles that use redirect names onto the canonical film page.
    """
    buf = []
    inside = False

    for line in lines:
        if "<page>" in line:
            inside = True
            buf = [line]
            continue

        if inside:
            buf.append(line)
            if "</page>" in line:
                page = "\n".join(buf)
                inside = False
                buf = []

                tm = TITLE_OPEN.search(page)
                if not tm:
                    continue
                title = tm.group(1)
                # skip non-mainspace pages (User:, File:, etc.)
                if ":" in title:
                    continue

                im = ID_OPEN.search(page)
                page_id = im.group(1) if im else None

                txm = TEXT_OPEN.search(page)
                if not txm:
                    continue
                t = txm.group(1)
                cxm = TEXT_CLOSE.search(t)
                text = (cxm.group(1) if cxm else t)

                rm = REDIRECT_RE.search(text)
                if not rm:
                    continue

                target = rm.group(1)

                yield {
                    "redirect_page_id": page_id,
                    "redirect_title": title,
                    "redirect_title_norm": normalize_title(title),
                    "target_title": target,
                    "target_title_norm": normalize_title(target),
                }


# ============================================================
#  Spark jobs
# ============================================================
def get_spark():
    py = sys.executable
    return (SparkSession.builder
            .appName("RT_Wiki_Join_CLI")
            .config("spark.hadoop.io.native.lib.available", "false")
            .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
            .config("spark.hadoop.fs.AbstractFileSystem.file.impl", "org.apache.hadoop.fs.local.LocalFs")
            .config("spark.pyspark.python", py)
            .config("spark.pyspark.driver.python", py)
            .config("spark.python.worker.faulthandler.enabled", "true")
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
            .config("spark.python.worker.reuse", "false")
            .getOrCreate())


def extract_rt_with_spark(sc, html_root: str):
    glob_path = os.path.join(html_root, "*.html")
    rdd = sc.wholeTextFiles(glob_path)

    def per_partition(pairs):
        extractor = RTDataExtractor()
        for path, html in pairs:
            try:
                meta = extractor.extract_from_html(html, path)
                title = meta.get("title")
                yield {
                    "file_path": meta.get("file_path") or path,
                    "page_type": meta.get("page_type"),
                    "url": meta.get("url"),
                    "title": title,
                    "title_norm": normalize_title(title),
                    "release_date": meta.get("release_date"),
                    "rt_year": year_from_rt_release(meta.get("release_date")),
                    "director": meta.get("director"),
                    "cast": meta.get("cast", []),
                    "genres": meta.get("genres", []),
                    "tomatometer": int(meta.get("tomatometer", 0) or 0),
                    "audience_score": int(meta.get("audience_score", 0) or 0),
                    "synopsis": meta.get("synopsis"),
                    "description": meta.get("description"),
                    "rating": meta.get("rating"),
                    "runtime": meta.get("runtime"),
                }
            except Exception:
                continue

    return rdd.mapPartitions(per_partition).toDF()


def _read_wiki_files_partition(paths_iter: Iterator[str]) -> Iterator[Dict[str, Any]]:
    for path in paths_iter:
        try:
            with bz2.open(path, "rb") as raw:
                text_stream = TextIOWrapper(raw, encoding="utf-8", errors="ignore")
                for rec in parse_wiki_pages_partition(text_stream):
                    yield rec
        except Exception:
            continue


def parse_wiki_with_spark(sc, wiki_dir: str):
    # read all multistream parts (*.bz2) as text lines (Spark can read bz2)
    pattern = os.path.join(wiki_dir, "enwiki-latest-pages-articles-multistream*.xml-*.bz2")
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

    lines = sc.textFile(pattern, minPartitions=max(2, (os.cpu_count() or 2)))

    # --- 1) canonical film pages (Infobox film) ---
    pages = lines.mapPartitions(parse_wiki_pages_partition).map(coerce_wiki)
    wiki_df_base = spark.createDataFrame(pages, schema=WIKI_SCHEMA)

    # --- 2) redirects (alternative titles) ---
    redirects = lines.mapPartitions(parse_wiki_redirects_partition)
    redirect_df = spark.createDataFrame(redirects)  # schema inferred: redirect_* + target_*

    # --- 3) map redirects onto film pages via normalized target title ---
    redirect_join = (
        redirect_df
        .join(
            wiki_df_base,
            redirect_df["target_title_norm"] == wiki_df_base["wiki_title_norm"],
            "inner",
        )
        .select(
            # keep canonical page_id (so all aliases point to same film)
            wiki_df_base["wiki_page_id"],
            # but use redirect title as another possible wiki_title / wiki_title_norm
            redirect_df["redirect_title"].alias("wiki_title"),
            redirect_df["redirect_title_norm"].alias("wiki_title_norm"),
            wiki_df_base["wiki_year"],
            wiki_df_base["wiki_directors"],
            wiki_df_base["wiki_starring"],
            wiki_df_base["wiki_languages"],
            wiki_df_base["wiki_countries"],
            wiki_df_base["wiki_released_raw"],
            wiki_df_base["wiki_runtime_minutes"],
            wiki_df_base["wiki_budget"],
            wiki_df_base["wiki_box_office"],
        )
    )

    # --- 4) union canonical titles + redirect aliases ---
    wiki_df = (
        wiki_df_base
        .unionByName(redirect_join)
        # optional: de-dup by (title_norm, year) to keep things sane
        .dropDuplicates(["wiki_title_norm", "wiki_year"])
    )

    print("Wiki schema:", wiki_df.schema.simpleString())
    return wiki_df


def join_rt_wiki(rt_df, wiki_df, out_dir: str, sample_rows: int = 300):
    from pyspark.sql import functions as F
    from pyspark.sql import Window
    import os

    JOIN_PARQUET = os.path.join(out_dir, "rt_wiki_join.parquet")
    JOIN_SAMPLE_DIR = os.path.join(out_dir, "rt_wiki_join_sample_csv")

    # Only movie / TV pages from RT
    rt_df = rt_df.where(F.col("page_type").isin("movie", "tv")).persist()

    # --- common year condition (disallow conflicting non-null years) ---
    cond_year_ok = (
        (rt_df["rt_year"].isNull() & wiki_df["wiki_year"].isNull()) |
        (rt_df["rt_year"].isNull() & wiki_df["wiki_year"].isNotNull()) |
        (rt_df["rt_year"].isNotNull() & wiki_df["wiki_year"].isNull()) |
        (rt_df["rt_year"] == wiki_df["wiki_year"])
    )

    # ============================================================
    # 1) EXACT JOIN: normalized title equality (+ year sanity)
    # ============================================================
    cond_title = rt_df["title_norm"] == wiki_df["wiki_title_norm"]

    joined_exact = (
        rt_df.join(wiki_df, cond_title & cond_year_ok, "inner")
             .withColumn(
                 "join_confidence",
                 F.when(
                     (F.col("rt_year").isNotNull()) &
                     (F.col("wiki_year").isNotNull()) &
                     (F.col("rt_year") == F.col("wiki_year")), F.lit(2)
                 ).otherwise(F.lit(1))
             )
    ).persist()

    # RT rows that didn't match exactly
    rt_unmatched = (
        rt_df.alias("rt")
             .join(
                 joined_exact.select("file_path").distinct().alias("j"),
                 on="file_path",
                 how="left_anti",
             )
    )

    # ============================================================
    # 2) FUZZY JOIN: same head+tail token + similar title + sane year
    # ============================================================

    def add_head_tail(df, title_col, head_col, tail_col):
        tokens = F.split(F.col(title_col), " ")
        return (
            df
            .withColumn(head_col, F.element_at(tokens, 1))
            .withColumn(tail_col, F.element_at(tokens, -1))
        )

    # add head / tail tokens for blocking
    rt_block = add_head_tail(rt_unmatched, "title_norm", "rt_head", "rt_tail")
    wiki_block = add_head_tail(wiki_df, "wiki_title_norm", "wiki_head", "wiki_tail")

    cond_year_fuzzy = (
        (rt_block["rt_year"].isNull() & wiki_block["wiki_year"].isNull()) |
        (rt_block["rt_year"].isNull() & wiki_block["wiki_year"].isNotNull()) |
        (rt_block["rt_year"].isNotNull() & wiki_block["wiki_year"].isNull()) |
        (rt_block["rt_year"] == wiki_block["wiki_year"])
    )

    # candidate pairs: same head+tail token, year sane
    cand = (
        rt_block.join(
            wiki_block,
            (rt_block.rt_head == wiki_block.wiki_head) &
            (rt_block.rt_tail == wiki_block.wiki_tail) &
            cond_year_fuzzy,
            "inner",
        )
        .withColumn("lev", F.levenshtein("title_norm", "wiki_title_norm"))
    )

    # take best (lowest distance) per RT file_path, with distance <= 3
    w = Window.partitionBy("file_path").orderBy(F.col("lev").asc())

    joined_fuzzy = (
        cand
        .withColumn("rank", F.row_number().over(w))
        .filter((F.col("rank") == 1) & (F.col("lev") <= 3))
        .drop("rank", "rt_head", "rt_tail", "wiki_head", "wiki_tail", "lev")
        .withColumn(
            "join_confidence",
            F.when(
                (F.col("rt_year").isNotNull()) &
                (F.col("wiki_year").isNotNull()) &
                (F.col("rt_year") == F.col("wiki_year")), F.lit(2)
            ).otherwise(F.lit(1))
        )
    )

    # ============================================================
    # 3) UNION exact + fuzzy, then save / report
    # ============================================================
    joined_all = (
        joined_exact
        .unionByName(joined_fuzzy)
        .dropDuplicates(["file_path", "wiki_page_id"])
    ).persist()

    # Save full join as Parquet (schema preserved, arrays OK)
    joined_all.write.mode("overwrite").parquet(JOIN_PARQUET)

    # ---- CSV sample: stringify array columns ----
    array_cols = [
        "genres", "cast",
        "wiki_directors", "wiki_starring", "wiki_languages", "wiki_countries"
    ]
    j2 = joined_all
    for c in array_cols:
        if c in j2.columns:
            j2 = j2.withColumn(
                c,
                F.concat_ws("|", F.when(F.col(c).isNull(), F.array(F.lit(""))).otherwise(F.col(c)))
            )

    cols = [
        "title", "rt_year", "genres", "director", "cast", "tomatometer", "audience_score",
        "wiki_title", "wiki_year", "wiki_directors", "wiki_starring", "wiki_languages",
        "wiki_countries", "wiki_runtime_minutes", "wiki_budget", "wiki_box_office", "join_confidence"
    ]
    cols = [c for c in cols if c in j2.columns]

    (j2.select(*cols)
       .limit(sample_rows)
       .coalesce(1)
       .write.mode("overwrite")
       .option("header", True)
       .option("nullValue", "")
       .csv(JOIN_SAMPLE_DIR))

    # ---- quick metrics ----
    total_rt = rt_df.select("file_path").distinct().count()
    total_wiki = wiki_df.select("wiki_page_id").distinct().count()
    connected_wiki = joined_all.select("wiki_page_id").distinct().count()
    joined_rows = joined_all.count()

    print("\n========== JOIN REPORT ==========")
    print(f"RT pages (movie/tv) parsed:      {total_rt:,}")
    print(f"Wiki film pages parsed:          {total_wiki:,}")
    print(f"JOINED rows (exact+fuzzy):       {joined_rows:,}")
    print(f"UNIQUE wiki pages connected:     {connected_wiki:,}")
    conf = (
        joined_all.groupBy("join_confidence")
                  .count()
                  .orderBy("join_confidence", ascending=False)
                  .collect()
    )
    print("\nJoin confidence breakdown:")
    for row in conf:
        level = {2: "title+year", 1: "title only"}.get(row["join_confidence"], str(row["join_confidence"]))
        print(f"  {level:12s} -> {row['count']:,} rows")
    print(f"\nSaved full join to: {JOIN_PARQUET}")
    print(f"CSV sample: {JOIN_SAMPLE_DIR}")


# ============================================================
#  Simple downloader for 'latest' multistream parts
# ============================================================
# --- replace list_latest_multistream_files and download_files with this ---
from html import unescape

try:
    import requests
except Exception:
    requests = None


def list_latest_multistream_files():
    """
    Returns absolute URLs for ALL enwiki 'pages-articles-multistream*.bz2' parts,
    plus the multistream index file.
    """
    if requests is None:
        print("requests not installed. Try: pip install requests tqdm")
        return []
    resp = requests.get(WIKI_LATEST_URL, timeout=60)
    resp.raise_for_status()
    html = resp.text
    # capture all multistream .bz2 file names (parts + index)
    names = re.findall(r'href="(enwiki-latest-pages-articles-multistream[^"]+?\.bz2)"', html)
    names = [unescape(n) for n in names]
    # de-dup and sort
    names = sorted(set(names))
    # prepend absolute
    return [WIKI_LATEST_URL + n for n in names]


def download_files(urls, out_dir, max_files=None):
    """
    Sequential download with a per-file progress bar (tqdm).
    Set max_files=None to download ALL. Resumes if file exists.
    """
    if requests is None:
        print("requests not installed. Try: pip install requests tqdm")
        return
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None
        print("tqdm not installed (no progress bars). pip install tqdm")

    os.makedirs(out_dir, exist_ok=True)
    todo = urls if max_files is None else urls[:max_files]

    for i, url in enumerate(todo, 1):
        local = os.path.join(out_dir, os.path.basename(url))
        if os.path.exists(local):
            print(f"[{i}/{len(todo)}] exists: {local}")
            continue

        print(f"[{i}/{len(todo)}] downloading: {url}")
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", "0"))
            chunk = 1024 * 1024  # 1 MB
            if tqdm:
                with open(local, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as bar:
                    for part in r.iter_content(chunk_size=chunk):
                        if part:
                            f.write(part)
                            bar.update(len(part))
            else:
                with open(local, "wb") as f:
                    for part in r.iter_content(chunk_size=chunk):
                        if part:
                            f.write(part)
        print(f"  -> saved: {local}")


# ============================================================
#  MENU
# ============================================================
def menu():
    global HTML_DIR, WIKI_DIR
    spark = get_spark()
    sc = spark.sparkContext
    print("Driver Python:", sys.version)
    print("Worker Python (conf):", get_spark().conf.get("spark.pyspark.python"))
    print("Hadoop version:",
          spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion())

    # wiki = spark.read.parquet(WIKI_PARQUET)
    # wiki.selectExpr(
    #     "count(*) as rows",
    #     "sum(case when size(wiki_directors) > 0 then 1 else 0 end) as with_directors",
    #     "sum(case when size(wiki_starring)  > 0 then 1 else 0 end) as with_starring",
    #     "sum(case when wiki_runtime_minutes is not null then 1 else 0 end) as with_runtime",
    #     "sum(case when wiki_budget is not null and wiki_budget <> '' then 1 else 0 end) as with_budget",
    #     "sum(case when wiki_box_office is not null and wiki_box_office <> '' then 1 else 0 end) as with_box_office"
    # ).show(truncate=False)

    while True:
        print("\n=== SPARK RT + WIKI MENU ===")
        print(f"Current paths:")
        print(f"  RT HTML dir : {HTML_DIR}")
        print(f"  Wiki dir    : {WIKI_DIR}")
        print(f"  Out dir     : {OUT_DIR}")
        print("1) Download a few Wikipedia multistream parts")
        print("2) write RT Parquet)")
        print("3) write Wiki Parquet)")
        print("4) Join RT and Wiki, write outputs & metrics")
        print("5) Change paths")
        print("6) Exit")
        choice = input("\nChoose (1-6): ").strip()

        if choice == "1":
            if requests is None:
                print("This helper needs 'requests'. Install with: pip install requests tqdm")
                continue
            try:
                urls = list_latest_multistream_files()
                print(f"Found {len(urls)} files (all multistream parts + index).")
                raw = input("How many files to download? (number or 'all') [3]: ").strip().lower() or "3"
                max_files = None if raw == "all" else int(raw)
                download_files(urls, WIKI_DIR, max_files=max_files)
                print("Done.")
            except Exception as e:
                print(f"Download error: {e}")

        elif choice == "2":
            try:
                print("\n=== Extracting RT HTML with Spark ===")
                rt_df = extract_rt_with_spark(sc, HTML_DIR).persist()
                rt_df.write.mode("overwrite").parquet(RT_PARQUET)
                total = rt_df.count()
                print(f"Saved RT parquet: {RT_PARQUET}  (rows: {total:,})")
            except Exception as e:
                print(f"RT extraction error: {e}")

        elif choice == "3":
            try:
                print("\n=== Parsing Wikipedia multistream dump with Spark (regex) ===")
                wiki_df = parse_wiki_with_spark(sc, WIKI_DIR).persist()
                wiki_df.write.mode("overwrite").parquet(WIKI_PARQUET)
                total = wiki_df.count()
                print(f"Saved Wiki parquet: {WIKI_PARQUET}  (rows: {total:,})")
            except Exception as e:
                print(
                    f"Wiki parse error: {e}\nHint: ensure files exist like {os.path.join(WIKI_DIR, 'enwiki-latest-pages-articles-multistream*.bz2')}")

        elif choice == "4":
            try:
                print("\n=== Join RT and Wiki ===")
                rt_df = spark.read.parquet(RT_PARQUET)
                wiki_df = spark.read.parquet(WIKI_PARQUET)
                wiki_df.printSchema()

                # Debug: show some wiki rows
                wiki_df.select(
                    "wiki_title",
                    "wiki_year",
                    "wiki_directors",
                    "wiki_starring",
                    "wiki_languages",
                    "wiki_countries",
                    "wiki_runtime_minutes",
                    "wiki_budget",
                    "wiki_box_office",
                ).show(20, truncate=False)

                join_rt_wiki(rt_df, wiki_df, OUT_DIR, sample_rows=300)
            except Exception as e:
                print(f"Join error: {e}\nMake sure {RT_PARQUET} and {WIKI_PARQUET} exist.")

        elif choice == "5":
            HTML_DIR = input(f"RT HTML dir [{HTML_DIR}]: ").strip() or HTML_DIR
            WIKI_DIR = input(f"Wiki dir    [{WIKI_DIR}]: ").strip() or WIKI_DIR

        elif choice == "6":
            print("Bye!");
            break
        else:
            print("Invalid choice.")

    spark.stop()


if __name__ == "__main__":
    menu()
