#!/usr/bin/env python
# rt_lucene_parquet_search.py
#
# Lucene-based index + CLI search for RottenTomatoes + Wikipedia joined data.
# Source: Spark output rt_wiki_join.parquet (exact+fuzzy RT/Wiki join).

import os
import time
from typing import Dict, Any, List, Optional, Tuple

import lucene
import pyarrow.parquet as pq

from java.nio.file import Paths

from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import (
    Document,
    Field,
    StringField,
    TextField,
    IntPoint,
    StoredField,
)
from org.apache.lucene.index import (
    IndexWriter,
    IndexWriterConfig,
    DirectoryReader,
    Term,
)
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.search import (
    IndexSearcher,
    BooleanQuery,
    BooleanClause,
    TermQuery,
    TopDocs,
    FuzzyQuery,
    BoostQuery,
)

from org.apache.lucene.queryparser.classic import MultiFieldQueryParser
from org.apache.lucene.queryparser.classic import QueryParser


# ============================================================
#  CONFIG / ENV
# ============================================================

# Path to Spark join parquet (directory OR single part-*.parquet file)
DEFAULT_PARQUET_PATH = os.environ.get("RT_WIKI_JOIN_PATH", "/data/rt_wiki_join.parquet")

# Directory where Lucene index lives
DEFAULT_INDEX_DIR = os.environ.get("RT_INDEX_DIR", "/data/rt_lucene_index")


# ============================================================
#  Lucene index wrapper
# ============================================================

class LuceneRTIndex:
    """
    Lucene-backed index for RT+Wiki joined records.

    We index:
      - Text fields:
          title, synopsis, description, genres_text, director, cast, wiki_title
      - Facet / filter fields:
          page_type, rating, director_exact, languages, countries
      - Numeric fields:
          release_year (rt_year or wiki_year),
          tomatometer, audience_score,
          runtime_minutes (wiki_runtime_minutes)
    """

    def __init__(self, index_dir: str = DEFAULT_INDEX_DIR):
        self.index_dir = index_dir
        self.analyzer: Optional[StandardAnalyzer] = None
        self.writer: Optional[IndexWriter] = None
        self.searcher: Optional[IndexSearcher] = None
        self.reader: Optional[DirectoryReader] = None
        self._vm_started = False

    # ---------- VM + directory management ----------

    def _ensure_vm(self):
        if not self._vm_started:
            lucene.initVM(vmargs=['-Djava.awt.headless=true'])
            self._vm_started = True

    def _get_directory(self):
        self._ensure_vm()
        path = Paths.get(self.index_dir)
        return MMapDirectory(path)

    # ---------- writer / searcher lifecycle ----------

    def open_writer(self, recreate: bool = True):
        self._ensure_vm()
        self.analyzer = StandardAnalyzer()
        directory = self._get_directory()
        config = IndexWriterConfig(self.analyzer)
        if recreate:
            config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        else:
            config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
        self.writer = IndexWriter(directory, config)

    def close_writer(self):
        if self.writer is not None:
            self.writer.commit()
            self.writer.close()
            self.writer = None

    def open_searcher(self):
        self._ensure_vm()
        directory = self._get_directory()
        self.reader = DirectoryReader.open(directory)
        self.searcher = IndexSearcher(self.reader)
        if self.analyzer is None:
            self.analyzer = StandardAnalyzer()

    def close_searcher(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
            self.searcher = None

    # ---------- helpers ----------

    def _add_multi_string_field(self, doc: Document, name: str, values):
        if not values:
            return
        if isinstance(values, list):
            for v in values:
                if v:
                    doc.add(StringField(name, str(v), Field.Store.YES))
        else:
            doc.add(StringField(name, str(values), Field.Store.YES))

    # ---------- add document from normalized meta dict ----------

    def add_document(self, meta: Dict[str, Any]):
        if self.writer is None:
            raise RuntimeError("IndexWriter is not open. Call open_writer() first.")

        doc = Document()

        # Stable doc id
        doc_id = (
            meta.get("file_path")
            or meta.get("wiki_page_id")
            or meta.get("url")
            or meta.get("doc_id")
            or ""
        )
        doc.add(StringField("doc_id", str(doc_id), Field.Store.YES))

        # Page type (movie / tv)
        page_type = (meta.get("page_type") or "").strip()
        doc.add(StringField("page_type", page_type, Field.Store.YES))

        # ---- TEXT FIELDS ----
        title = meta.get("title") or ""
        doc.add(TextField("title", title, Field.Store.YES))

        synopsis = meta.get("synopsis") or ""
        doc.add(TextField("synopsis", synopsis, Field.Store.YES))

        description = meta.get("description") or ""
        doc.add(TextField("description", description, Field.Store.YES))

        # Genres
        genres = meta.get("genres") or []
        self._add_multi_string_field(doc, "genres", genres)
        if genres:
            doc.add(TextField("genres_text", " ".join(genres), Field.Store.NO))

        # Director + exact director field
        director = meta.get("director") or ""
        if director:
            doc.add(TextField("director", director, Field.Store.YES))
            doc.add(StringField("director_exact", director, Field.Store.YES))

        # Cast (multi-valued text)
        cast = meta.get("cast") or []
        if cast:
            for name in cast:
                if name:
                    doc.add(TextField("cast", str(name), Field.Store.YES))

        # Rating (G/PG/PG-13/R/TV-14, etc.)
        rating = meta.get("rating") or ""
        if rating:
            doc.add(StringField("rating", rating, Field.Store.YES))

        # Wiki title for cross-checking / alternate search
        wiki_title = meta.get("wiki_title") or ""
        if wiki_title:
            doc.add(TextField("wiki_title", wiki_title, Field.Store.YES))

        wiki_title_norm = meta.get("wiki_title_norm") or ""
        if wiki_title_norm:
            doc.add(StringField("wiki_title_norm", wiki_title_norm, Field.Store.YES))

        # Languages / countries
        languages = meta.get("wiki_languages") or meta.get("languages") or []
        self._add_multi_string_field(doc, "languages", languages)

        countries = meta.get("wiki_countries") or meta.get("countries") or []
        self._add_multi_string_field(doc, "countries", countries)

        # ---- NUMERIC FIELDS ----

        # Year: prefer RT year, fall back to wiki year
        year = meta.get("rt_year")
        if year is None:
            year = meta.get("wiki_year")
        if year is not None:
            try:
                y = int(year)
                doc.add(IntPoint("release_year", y))
                doc.add(StoredField("release_year", y))
            except Exception:
                pass

        tom = meta.get("tomatometer")
        if tom is not None:
            try:
                t = int(tom)
                doc.add(IntPoint("tomatometer", t))
                doc.add(StoredField("tomatometer", t))
            except Exception:
                pass

        aud = meta.get("audience_score")
        if aud is not None:
            try:
                a = int(aud)
                doc.add(IntPoint("audience_score", a))
                doc.add(StoredField("audience_score", a))
            except Exception:
                pass

        runtime_min = meta.get("wiki_runtime_minutes")
        if runtime_min is not None:
            try:
                m = int(runtime_min)
                doc.add(IntPoint("runtime_minutes", m))
                doc.add(StoredField("runtime_minutes", m))
            except Exception:
                pass

        # join_confidence stored for debugging / analysis (2=title+year, 1=title)
        jc = meta.get("join_confidence")
        if jc is not None:
            try:
                jcv = int(jc)
                doc.add(IntPoint("join_confidence", jcv))
                doc.add(StoredField("join_confidence", jcv))
            except Exception:
                pass

        # Stored URL / file_path
        url = meta.get("url") or ""
        if url:
            doc.add(StoredField("url", url))

        file_path = meta.get("file_path") or ""
        if file_path:
            doc.add(StoredField("file_path", file_path))

        self.writer.addDocument(doc)

    # ---------- query construction ----------

    def _build_text_query(self, query_str: str):
        """
        Multi-field query: OR inside each field, OR across fields, with boosts.
        This mimics your old "bag-of-words over everything" behaviour while
        still letting BM25 do ranking.
        """
        fields = ["title", "synopsis", "genres_text", "director", "cast", "wiki_title"]
        boosts = {
            "title": 3.0,
            "synopsis": 2.0,
            "genres_text": 1.5,
            "director": 1.2,
            "cast": 1.5,
            "wiki_title": 1.5,
        }

        builder = BooleanQuery.Builder()

        for field in fields:
            parser = QueryParser(field, self.analyzer)
            parser.setDefaultOperator(QueryParser.Operator.OR)

            # You can escape special chars if you want to be extra safe
            q = parser.parse(query_str)

            boost = boosts.get(field, 1.0)
            if boost != 1.0:
                q = BoostQuery(q, boost)

            builder.add(q, BooleanClause.Occur.SHOULD)

        return builder.build()

    # ---------- search ----------

    def search(
        self,
        query_str: str,
        top_k: int = 10,
        page_type: Optional[str] = None,
        year_range: Optional[Tuple[int, int]] = None,
        tomatometer_min: Optional[int] = None,
        audience_min: Optional[int] = None,
        min_join_conf: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if self.searcher is None:
            self.open_searcher()

        main_q = self._build_text_query(query_str)

        builder = BooleanQuery.Builder()
        builder.add(main_q, BooleanClause.Occur.MUST)

        if page_type:
            builder.add(
                TermQuery(Term("page_type", page_type)),
                BooleanClause.Occur.FILTER,
            )

        if year_range is not None:
            lo, hi = year_range
            yr_q = IntPoint.newRangeQuery("release_year", lo, hi)
            builder.add(yr_q, BooleanClause.Occur.FILTER)

        if tomatometer_min is not None:
            q_tom = IntPoint.newRangeQuery("tomatometer", tomatometer_min, 100)
            builder.add(q_tom, BooleanClause.Occur.FILTER)

        if audience_min is not None:
            q_aud = IntPoint.newRangeQuery("audience_score", audience_min, 100)
            builder.add(q_aud, BooleanClause.Occur.FILTER)

        if min_join_conf is not None:
            q_jc = IntPoint.newRangeQuery("join_confidence", min_join_conf, 10)
            builder.add(q_jc, BooleanClause.Occur.FILTER)

        final_q = builder.build()
        top_docs: TopDocs = self.searcher.search(final_q, top_k)

        results: List[Dict[str, Any]] = []
        for score_doc in top_docs.scoreDocs:
            doc = self.searcher.doc(score_doc.doc)
            res: Dict[str, Any] = {
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title"),
                "page_type": doc.get("page_type"),
                "url": doc.get("url"),
                "file_path": doc.get("file_path"),
                "score": float(score_doc.score),
            }

            # Stored numeric stuff
            for fld in (
                    "release_year",
                    "tomatometer",
                    "audience_score",
                    "runtime_minutes",
                    "join_confidence",
            ):
                val = doc.get(fld)
                if val is not None:
                    res[fld] = val

            # Multi-valued fields: genres, cast
            try:
                genres_vals = doc.getValues("genres")
            except Exception:
                genres_vals = []
            if genres_vals:
                # convert Java array -> Python list
                res["genres"] = list(genres_vals)

            try:
                cast_vals = doc.getValues("cast")
            except Exception:
                cast_vals = []
            if cast_vals:
                res["cast"] = list(cast_vals)

            results.append(res)

        return results

    def fuzzy_title_search(self, title: str, max_edits: int = 2, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.searcher is None:
            self.open_searcher()

        # StandardAnalyzer lowercases tokens, so we do the same
        term = Term("title", title.lower())
        fq = FuzzyQuery(term, max_edits)
        top_docs = self.searcher.search(fq, top_k)
        out: List[Dict[str, Any]] = []
        for sd in top_docs.scoreDocs:
            doc = self.searcher.doc(sd.doc)
            out.append({
                "title": doc.get("title"),
                "page_type": doc.get("page_type"),
                "url": doc.get("url"),
                "score": float(sd.score),
            })
        return out


# ============================================================
#  Search engine: build index from rt_wiki_join.parquet
# ============================================================

class LuceneRTSearchEngine:
    def __init__(self, parquet_path: str = DEFAULT_PARQUET_PATH, index_dir: str = DEFAULT_INDEX_DIR):
        self.parquet_path = parquet_path
        self.index = LuceneRTIndex(index_dir=index_dir)
        self.loaded = False

    # ---------- build index from parquet ----------

    def build_index(
        self,
        max_docs: Optional[int] = None,
        min_join_conf: int = 1,
    ) -> bool:
        path = self.parquet_path
        if not os.path.exists(path):
            print(f"Error: parquet path '{path}' does not exist in container.")
            return False

        print(f"Reading rt_wiki_join parquet from: {path}")
        table = pq.read_table(path)
        rows = table.to_pylist()
        total_rows = len(rows)
        print(f"Loaded {total_rows} joined rows from parquet.")

        self.index.open_writer(recreate=True)

        count = 0
        for row in rows:
            # Spark join schema columns; we normalize them into 'meta'
            join_conf = row.get("join_confidence")
            if join_conf is not None and join_conf < min_join_conf:
                # 1 or 2 in your join; you can choose to only index 2 if you want.
                pass

            meta: Dict[str, Any] = {
                # RT side
                "file_path": row.get("file_path"),
                "page_type": row.get("page_type"),
                "url": row.get("url"),
                "title": row.get("title"),
                "title_norm": row.get("title_norm"),
                "description": row.get("description"),
                "synopsis": row.get("synopsis"),
                "genres": row.get("genres"),
                "director": row.get("director"),
                "cast": row.get("cast"),
                "rating": row.get("rating"),
                "rt_year": row.get("rt_year"),
                "tomatometer": row.get("tomatometer"),
                "audience_score": row.get("audience_score"),
                # Wiki side
                "wiki_page_id": row.get("wiki_page_id"),
                "wiki_title": row.get("wiki_title"),
                "wiki_title_norm": row.get("wiki_title_norm"),
                "wiki_year": row.get("wiki_year"),
                "wiki_directors": row.get("wiki_directors"),
                "wiki_starring": row.get("wiki_starring"),
                "wiki_languages": row.get("wiki_languages"),
                "wiki_countries": row.get("wiki_countries"),
                "wiki_runtime_minutes": row.get("wiki_runtime_minutes"),
                "wiki_budget": row.get("wiki_budget"),
                "wiki_box_office": row.get("wiki_box_office"),
                "join_confidence": join_conf,
            }

            # Fallbacks: if RT director/cast missing, use wiki side
            if not meta.get("director") and meta.get("wiki_directors"):
                meta["director"] = ", ".join(meta["wiki_directors"])
            if (not meta.get("cast")) and meta.get("wiki_starring"):
                meta["cast"] = meta["wiki_starring"]

            try:
                self.index.add_document(meta)
                count += 1
                if max_docs is not None and count >= max_docs:
                    break
                if count % 5000 == 0:
                    print(f"  indexed {count} docs...")
            except Exception as e:
                print(f"Error indexing row with file_path={meta.get('file_path')}: {e}")

        self.index.close_writer()
        self.loaded = True
        print(f"Lucene indexing complete. Indexed {count} documents (from {total_rows} rows).")
        return True

    # ---------- search wrappers ----------

    def _ensure_loaded(self):
        if not self.loaded:
            try:
                self.index.open_searcher()
                self.loaded = True
            except Exception as e:
                raise RuntimeError(f"Lucene index not built or cannot be opened: {e}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        page_type: Optional[str] = None,
        year_range: Optional[Tuple[int, int]] = None,
        tomatometer_min: Optional[int] = None,
        audience_min: Optional[int] = None,
        min_join_conf: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        return self.index.search(
            query_str=query,
            top_k=top_k,
            page_type=page_type,
            year_range=year_range,
            tomatometer_min=tomatometer_min,
            audience_min=audience_min,
            min_join_conf=min_join_conf,
        )

    def fuzzy_title_search(self, query: str, max_edits: int = 2, top_k: int = 10) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        return self.index.fuzzy_title_search(query, max_edits=max_edits, top_k=top_k)


# ============================================================
#  CLI
# ============================================================

class SearchCLI:
    def __init__(self, parquet_path: str = DEFAULT_PARQUET_PATH, index_dir: str = DEFAULT_INDEX_DIR):
        self.engine = LuceneRTSearchEngine(parquet_path=parquet_path, index_dir=index_dir)
        self.parquet_path = parquet_path
        self.index_dir = index_dir
        self.setup_complete = False

    # ---- console helpers ----

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        print("\n" + "=" * 70)
        print("       RT + WIKI LUCENE SEARCH (BM25, PyLucene, Parquet)")
        print("=" * 70)
        print(f"Parquet : {self.parquet_path}")
        print(f"Index   : {self.index_dir}")

    def print_menu(self):
        print("\nMAIN MENU:")
        print("1. Build Lucene index from rt_wiki_join.parquet")
        print("2. Open existing Lucene index (no rebuild)")
        print("3. Search (multi-field BM25 + filters)")
        print("4. Fuzzy title search (typo-tolerant)")
        print("0. Exit")
        print("-" * 70)

    def wait_for_enter(self):
        input("\nPress Enter to continue...")

    # ---- setup ----

    def setup_engine(self):
        self.clear_screen()
        self.print_header()

        print("\nSETUP LUCENE ENGINE")
        print("1. Build new Lucene index from parquet")
        print("2. Assume existing Lucene index on disk")
        print("3. Back to main menu")

        choice = input("\nChoose option (1-3): ").strip()

        if choice == "1":
            max_docs_raw = input("Max documents to index (Enter for all): ").strip()
            max_docs = int(max_docs_raw) if max_docs_raw.isdigit() else None

            min_jc_raw = input("Min join_confidence [1 or 2, default 1]: ").strip()
            try:
                min_jc = int(min_jc_raw) if min_jc_raw else 1
            except Exception:
                min_jc = 1

            if self.engine.build_index(max_docs=max_docs, min_join_conf=min_jc):
                self.setup_complete = True
            else:
                print("Failed to build Lucene index")
            self.wait_for_enter()

        elif choice == "2":
            try:
                self.engine.index.open_searcher()
                self.engine.loaded = True
                self.setup_complete = True
                print("Lucene index opened successfully!")
            except Exception as e:
                print(f"Failed to open index: {e}")
            self.wait_for_enter()

    # ---- search ----

    def do_search(self):
        if not self.setup_complete:
            print("Index not set up yet. Use option 1 or 2 first.")
            self.wait_for_enter()
            return

        self.clear_screen()
        self.print_header()

        print("\nSEARCH (Lucene BM25, multi-field)")
        query = input("Enter search query: ").strip()
        if not query:
            print("No query entered!")
            self.wait_for_enter()
            return

        page_type = input("Filter page_type [movie/tv/empty for all]: ").strip() or None

        yr_raw = input("Year range (e.g. 1990-1999, empty for none): ").strip()
        year_range: Optional[Tuple[int, int]] = None
        if yr_raw:
            try:
                lo, hi = yr_raw.split("-")
                year_range = (int(lo), int(hi))
            except Exception:
                print("Invalid year range, ignoring.")

        tom_min_raw = input("Min Tomatometer (0-100, empty for none): ").strip()
        tom_min = int(tom_min_raw) if tom_min_raw.isdigit() else None

        aud_min_raw = input("Min Audience score (0-100, empty for none): ").strip()
        aud_min = int(aud_min_raw) if aud_min_raw.isdigit() else None

        jc_min_raw = input("Min join_confidence [1/2, empty for none]: ").strip()
        jc_min = int(jc_min_raw) if jc_min_raw.isdigit() else None

        try:
            start_time = time.time()
            results = self.engine.search(
                query=query,
                top_k=10,
                page_type=page_type,
                year_range=year_range,
                tomatometer_min=tom_min,
                audience_min=aud_min,
                min_join_conf=jc_min,
            )
            query_time = time.time() - start_time
            self.display_results(results, query, "Lucene BM25", query_time)
        except Exception as e:
            print(f"Search error: {e}")

        self.wait_for_enter()

    def do_fuzzy_search(self):
        if not self.setup_complete:
            print("Index not set up yet. Use option 1 or 2 first.")
            self.wait_for_enter()
            return

        self.clear_screen()
        self.print_header()

        print("\nFUZZY TITLE SEARCH")
        query = input("Enter (possibly misspelled) title: ").strip()
        if not query:
            print("No query entered!")
            self.wait_for_enter()
            return

        edits_raw = input("Max Levenshtein edits [1-2, default 2]: ").strip()
        try:
            max_edits = int(edits_raw) if edits_raw else 2
        except Exception:
            max_edits = 2

        try:
            start_time = time.time()
            results = self.engine.fuzzy_title_search(query, max_edits=max_edits, top_k=10)
            query_time = time.time() - start_time
            self.display_results(results, query, f"Fuzzy title (edits={max_edits})", query_time)
        except Exception as e:
            print(f"Fuzzy search error: {e}")

        self.wait_for_enter()

    # ---- display ----

    def display_results(self, results: List[Dict[str, Any]], query: str, method: str, query_time: float):
        if not results:
            print(f"\n{method} - No results for '{query}' (time {query_time:.3f}s).")
            return

        print(f"\n{method} - Found {len(results)} results for '{query}' in {query_time:.3f}s")
        print("=" * 80)

        for i, res in enumerate(results, 1):
            title = res.get("title") or "No Title"
            page_type = res.get("page_type") or "unknown"
            score = res.get("score", 0.0)
            print(f"\n{i}. {title}")
            print(f"   Type: {page_type}")
            print(f"   Score: {score:.4f}")

            url = res.get("url")
            if url:
                print(f"   URL: {url}")
            fp = res.get("file_path")
            if fp:
                print(f"   File: {fp}")

            ry = res.get("release_year")
            if ry:
                print(f"   Year: {ry}")

            tom = res.get("tomatometer")
            if tom is not None:
                print(f"   Tomatometer: {tom}%")
            aud = res.get("audience_score")
            if aud is not None:
                print(f"   Audience score: {aud}%")

            rt = res.get("runtime_minutes")
            if rt is not None:
                print(f"   Runtime: {rt} min")

            jc = res.get("join_confidence")
            if jc is not None:
                print(f"   Join confidence: {jc}")

            # New: genres + cast
            genres = res.get("genres")
            if genres:
                print(f"   Genres: {', '.join(genres)}")

            cast = res.get("cast")
            if cast:
                # limit for readability, but still useful
                cast_preview = cast[:6]
                more = len(cast) - len(cast_preview)
                if more > 0:
                    print(f"   Cast: {', '.join(cast_preview)} (+{more} more)")
                else:
                    print(f"   Cast: {', '.join(cast_preview)}")

            print("-" * 40)

    # ---- main loop ----

    def run(self):
        while True:
            self.clear_screen()
            self.print_header()
            print(f"\nIndex status: {'READY' if self.setup_complete else 'NOT LOADED'}")
            self.print_menu()

            choice = input("Enter your choice (0-4): ").strip()

            if choice == "0":
                print("\nExiting ...")
                break
            elif choice == "1":
                self.setup_engine()
            elif choice == "2":
                self.setup_engine()
            elif choice == "3":
                self.do_search()
            elif choice == "4":
                self.do_fuzzy_search()
            else:
                print("Invalid choice! Try again.")
                self.wait_for_enter()


# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    parquet_path = os.environ.get("RT_WIKI_JOIN_PATH", DEFAULT_PARQUET_PATH)
    index_dir = os.environ.get("RT_INDEX_DIR", DEFAULT_INDEX_DIR)

    cli = SearchCLI(parquet_path=parquet_path, index_dir=index_dir)
    cli.run()
