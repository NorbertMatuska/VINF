# Rotten Tomatoes + Wikipedia Movie Pipeline

End-to-end pipeline for:

1. Crawling Rotten Tomatoes HTML
2. Extracting + joining with Wikipedia film infobox data via Spark
3. Building a Lucene BM25 search index (inside Docker)

All pieces are wired around simple files and Parquet, no DB.

---

## Project structure

```text
crawler.py         # Rotten Tomatoes CSV-based crawler
extractor.py       # Spark pipeline: RT HTML -> RT parquet, Wiki dumps -> parquet, join -> rt_wiki_join.parquet
indexer_lucene.py  # PyLucene BM25 index + CLI over rt_wiki_join.parquet
Dockerfile         # Docker image for Lucene search (PyLucene + PyArrow)
old_extractor.py   # Legacy RT HTML extractor + pipeline (obsolete)
old_indexer.py     # Legacy Python inverted index search (obsolete)
requirements.txt   # Python dependencies for crawler + Spark extractor
```

## 1. `crawler.py` Rotten Tomatoes CSV Crawler

This module implements a domain-specific web crawler for Rotten Tomatoes that stores its full state in CSV files instead of a database. It is designed for long-running crawls that can be paused and resumed safely.

### Key features

- **Domain-restricted crawling**: Only crawls `rottentomatoes.com` movie pages (`/m/` paths), with aggressive URL normalization to avoid duplicates.
- **CSV-based state management**:
  - `csv_data/urls.csv` – URL queue with status (`discovered`, `queued`, `crawled`, `failed`), depth, timestamps, response codes, and file paths.
  - `csv_data/crawl_stats.csv` – per-run statistics (placeholder for higher-level analytics).
  - `csv_data/crawl_checkpoints.csv` – simple checkpointing for resume.
- **Robots.txt aware**: Fetches and caches `robots.txt` per domain, respects `crawl-delay`, and blocks disallowed URLs.
- **Rate limiting & anti-ban logic**:
  - Tracks requests per minute and per hour.
  - Adds jittered delays and exponential backoff for server-side errors and `429 Too Many Requests`.
  - Rotates realistic desktop user-agents periodically.
- **Robust error handling**:
  - Differentiates 404/4xx from real blocking errors.
  - Marks URLs as failed with error messages and retry counts in CSV.
- **HTML storage & link extraction**:
  - Saves each crawled page as an HTML file in `rotten_tomatoes_data/html_pages`.
  - Extracts new links from `<a>`, `data-href`, and `data-url` attributes with basic filtering.
- **Resume support**:
  - Detects interrupted runs and resets unfinished `queued`/`failed` URLs (under a retry limit) back to `discovered`.
  - Periodically writes checkpoints and prints progress, ETA, and basic crawl statistics.

### Running the crawler

```bash
python crawler.py
```

## 2. `extractor.py` RT + Wikipedia Spark Extractor

This module builds a joined movie dataset by combining **Rotten Tomatoes HTML dumps** with **Wikipedia film infoboxes** using Apache Spark.

---

### Prerequisites

You **must** have the following installed and working:

- **Python:** 3.11 (the script uses `sys.executable` for Spark workers)
- **Java:** 17
- **Hadoop:** 3.4.1, installed on `C:\` with binaries under `C:\hadoop\bin`

The script sets these environment variables on startup:

- `HADOOP_HOME = C:\hadoop`
- `hadoop.home.dir = C:\hadoop`
- `PATH` is extended with `C:\hadoop\bin`
- `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` are both set to `sys.executable`

You also need a working local Spark setup (`pyspark`), running in local mode is fine.

---

### What it does

- **RT HTML to Parquet**
  - Scans a directory of Rotten Tomatoes HTML files (`HTML_DIR`, default: `rt_roots_html`).
  - Uses `RTDataExtractor` (regex + JSON islands) to pull:
    - page type (movie/tv)
    - title + normalized title
    - release date and inferred year
    - genres, director, cast
    - Tomatometer & audience score
    - rating, runtime, synopsis, description
  - Writes `rt.parquet` into `OUT_DIR`.

- **Wikipedia film infobox to Parquet**
  - Reads compressed multistream dumps from `WIKI_DIR`
    (`enwiki-latest-pages-articles-multistream*.xml-*.bz2`).
  - Parses pages via regex and extracts from `{{Infobox film}}`:
    - title + normalized title
    - year, directors, starring, languages, countries
    - runtime (minutes), budget, box office
  - Also parses **redirect pages** to map alternative titles to canonical film pages.
  - Writes `wiki.parquet` into `OUT_DIR` with a fixed schema.

- **RT–Wiki join**
  - Filters RT to movie/tv pages.
  - **Exact join**: normalized RT title = normalized Wiki title, with year sanity checks.
  - **Fuzzy join**:
    - same head+tail tokens,
    - Levenshtein distance =< 3,
    - year-compatible.
  - Produces:
    - `rt_wiki_join.parquet` – full joined dataset.
    - `rt_wiki_join_sample_csv/` – small CSV sample with array columns stringified.
  - Prints a join report with counts and join-confidence breakdown.

---

### Downloading Wikipedia dumps

This script includes a **built-in downloader**:

- `list_latest_multistream_files()` lists all
  `enwiki-latest-pages-articles-multistream*.bz2` URLs from the “latest” dump.
- `download_files(...)` downloads them to `WIKI_DIR` with optional `tqdm` progress bars.

However, this helper is **sequential and relatively slow**.

For large dumps it’s **much faster** to use a CLI download tool that supports multiple
connections and parallel downloads, such as:

- `aria2c` – e.g.:

  ```bash
  aria2c -x 16 -s 16 -d D:\ProgrammingFun\VINFData\wiki_dumps\enwiki_latest \
    "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream1.xml-p1p41242.bz2"

Pull the full list of dump URLs (from the dump index page or `list_latest_multistream_files`)
and feed them into `aria2c` (or similar). The built-in Python downloader is mainly there as a
fallback, not the optimal path.

---

### Usage

Run the interactive menu:

```bash
python extractor.py
```

Menu options:

1. Download a few Wikipedia multistream parts (Small helper, for first time testing).
2. Extract Rotten Tomatoes HTML.
3. Parse Wikipedia dumps.
4. Join RT and Wiki, write parquet + CSV sample, and print metrics.
5. Change input/output paths.
6. Exit (shuts down Spark).

Make sure the RT HTML directory, Wiki dump directory and output directory in the config section
match your environment.

---

## 3. `indexer_lucene.py` RT + Wiki Lucene Search (Dockerized)

This component provides an **interactive Lucene/BM25 search CLI** running entirely inside a Docker container based on the prebuilt image `coady/pylucene:9.12.0`.  

The container loads:

- **PyLucene** (Lucene 9.12)
- **PyArrow** to read the Spark parquet join file
- The search engine (`indexer_lucene.py`)

It mounts two things:

1. The **Spark join parquet**:  
   `rt_wiki_join.parquet`  -> `/data/rt_wiki_join.parquet`

2. An **index directory** for Lucene to write into:  
   host path -> `/data/rt_lucene_index`

The entrypoint runs `indexer_lucene.py`, which launches an interactive CLI.

---

### What this tool does

Inside the container, the CLI lets you:

#### **1. Build a Lucene index** from the parquet file  
It indexes:

- Text fields:  
  `title`, `synopsis`, `description`, `genres_text`, `director`, `cast`, `wiki_title`
- Exact/filter fields:  
  `page_type`, `rating`, `director_exact`, `languages`, `countries`
- Numeric fields:  
  `release_year`, `tomatometer`, `audience_score`, `runtime_minutes`, `join_confidence`

It uses **StandardAnalyzer + BM25**.

#### **2. Re-open an existing index**  
(If you already built it and just want to query.)

#### **3. Run BM25 multi-field searches**  
Query flows through OR-operator QueryParsers across multiple fields with boosts.

#### **4. Run fuzzy title search**  
Typo-tolerant lookup using Lucene’s `FuzzyQuery`.

### Usage

#### 1. Build the image

```bash
docker build -t rt-lucene-search .
```

#### 2. Run the container and mount your files

(PowerShell syntax — adjust for Bash if needed.)

```powershell
docker run --rm -it `
  -v "D:\ProgrammingFun\VINFData\spark_out\rt_wiki_join.parquet:/data/rt_wiki_join.parquet" `
  -v "D:\ProgrammingFun\VINFData\rt_lucene_index:/data/rt_lucene_index" `
  rt-lucene-search
```

Inside the container you’ll see the interactive search UI:

```
1. Build Lucene index from rt_wiki_join.parquet
2. Open existing Lucene index
3. Search (BM25)
4. Fuzzy title search
0. Exit
```

If you mount trash paths or non-existent parquet files, the container will obviously tell you.

---

### Notes

* All heavy lifting (JVM, Lucene, PyLucene) is handled by the container.
* Your host OS needs *nothing* except Docker.
* Index is persistent only if you mount it.
  (Otherwise it dies with the container — obviously.)

---

## 4. `old_indexer.py` Legacy Python inverted index (obsolete)

There is an older, pure-Python implementation of the search engine (`InvertedIndex` + `RTSearchEngine` + `SearchMenu`) that:

* builds an in-memory inverted index from JSONL metadata in `extracted_metadata/`,
* supports multiple custom IDF variants and a homegrown BM25,
* saves/loads the index from a JSON file (`search_index.json`).

This implementation is now **obsolete** and kept only for reference/experiments.
The production path is the **PyLucene-based Dockerized indexer and search CLI** described above.

---

## 5. `old_extractor.py` Legacy Rotten Tomatoes HTML extractor (obsolete)

There is an older Rotten Tomatoes **HTML extractor + pipeline** that:

- parses raw RT HTML files with a regex-first `RTDataExtractor` (JSON islands + light HTML fallbacks),
- runs a batch `RTExtractionPipeline` over `rotten_tomatoes_data/html_pages`,
- writes metadata to `extracted_metadata/` as JSONL/CSV/Pickle,
- includes statistics (`RTStatistics`) and a small unit-test harness (`RTUnitTests`).

This stack is now **obsolete**, *slow* and kept only for historical reference and experiments on raw HTML dumps.  
The maintained path is the newer Spark-based processing + PyLucene indexing pipeline.

