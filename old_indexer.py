import json
import os
import re
import math
import time
import unicodedata
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Dict[str, float]] = {}            # term -> {doc_id: weighted_tf}
        self.documents: Dict[str, Dict[str, Any]] = {}          # doc_id -> raw doc
        self.doc_lengths: Dict[str, int] = {}                   # doc_id -> token count (post-stopword)
        self.doc_term_freqs: Dict[str, Counter] = {}            # doc_id -> Counter(term -> weighted_tf)
        self.doc_count = 0
        self.avg_doc_length = 0.0

        # tiny stoplist
        self._stop = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        # keep these short tokens
        self._whitelist_short = {'tvma', 'pg13', 'r', 'tv14', 'tvpg', 'ma', 'pg'}

        # default field weights
        self._field_weights = {
            'title': 3.0,
            'synopsis': 2.0,
            'description': 0.2,   # downweight boilerplate :)
            'genres': 1.3,
            'director': 1.2,
            'cast': 1.5,
            'rating': 1.1,
        }

    # --------- utils ---------
    def _normalize(self, text: str) -> str:
        # strip accents and lowercase
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower()

    def _tokenize(self, text: str) -> List[str]:
        text = self._normalize(text)
        text = re.sub(r'[^\w\s-]', ' ', text)        # keep hyphens for rating glue
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = []
        for tok in text.split():
            # join ratings like "tv-ma" -> "tvma"
            if '-' in tok and any(part.isalpha() for part in tok.split('-')):
                glued = tok.replace('-', '')
                if 1 <= len(glued) <= 4:
                    tok = glued
            # general filter
            if (tok not in self._stop) and (len(tok) > 2 or tok in self._whitelist_short):
                tokens.append(tok)
        return tokens

    # --------- core API ---------
    def add_document(self, doc_id: str, document: Dict[str, Any], fields: List[str] = None,
                     field_weights: Dict[str, float] = None):
        if fields is None:
            fields = ["title", "description", "synopsis", "genres", "director", "cast", "rating"]
        if field_weights is None:
            field_weights = self._field_weights

        # if doc already exists, remove its postings cleanly first (avoids index corruption)
        if doc_id in self.documents:
            self._remove_document(doc_id)

        # assemble weighted term counts per field
        per_field_counts: Counter = Counter()
        total_tokens = 0

        for field in fields:
            if field not in document:
                continue
            weight = field_weights.get(field, 1.0)
            val = document[field]
            if isinstance(val, list):
                field_text = ' '.join(str(x) for x in val)
            else:
                field_text = str(val)
            toks = self._tokenize(field_text)
            total_tokens += len(toks)
            if weight != 1.0:
                per_field_counts.update({t: weight for t in toks})
            else:
                per_field_counts.update(toks)

        # register doc
        self.documents[doc_id] = document
        self.doc_term_freqs[doc_id] = per_field_counts
        self.doc_lengths[doc_id] = total_tokens

        # update postings
        for term, wtf in per_field_counts.items():
            bucket = self.index.setdefault(term, {})
            bucket[doc_id] = float(wtf)

        # maintain derived stats
        self.doc_count = len(self.documents)
        total_len = sum(self.doc_lengths.values()) or 1
        self.avg_doc_length = total_len / self.doc_count

    def _remove_document(self, doc_id: str):
        # remove old postings for this doc to avoid index corruption
        old = self.doc_term_freqs.get(doc_id)
        if old:
            for term in list(old.keys()):
                postings = self.index.get(term)
                if postings and doc_id in postings:
                    del postings[doc_id]
                    if not postings:
                        del self.index[term]
        # drop meta
        self.doc_term_freqs.pop(doc_id, None)
        self.doc_lengths.pop(doc_id, None)
        self.documents.pop(doc_id, None)

    # --- IDFs ---
    def _calculate_idf_standard(self, term: str) -> float:
        # classic tf-idf idf = ln(N/df)
        df = len(self.index.get(term, {}))
        if df == 0:
            return 0.0
        return math.log(self.doc_count / df)

    def _calculate_idf_probabilistic(self, term: str) -> float:
        # BM25 idf clamped at 0 to avoid negatives
        df = len(self.index.get(term, {}))
        if df == 0:
            return 0.0
        val = math.log((self.doc_count - df + 0.5) / (df + 0.5))
        return max(0.0, val)

    def _calculate_idf_smooth(self, term: str) -> float:
        # smoothed variant
        df = len(self.index.get(term, {}))
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def _calculate_idf_max(self, term: str) -> float:
        # relative to max df
        df = len(self.index.get(term, {}))
        if df == 0:
            return 0.0
        max_df = max((len(v) for v in self.index.values()), default=1)
        return math.log(1.0 + max_df / (df + 1.0))

    def search(self, query: str, top_k: int = 10,
               idf_method: str = "standard",
               use_bm25: bool = False,
               bm25_params: Dict[str, float] = None,
               filters: Dict[str, Any] = None) -> List[Tuple[str, float]]:

        if bm25_params is None:
            bm25_params = {"k1": 1.5, "b": 0.75}
        if filters is None:
            filters = {}

        q_terms = self._tokenize(query)
        scores = defaultdict(float)

        # choose IDF
        if use_bm25:
            idf_fn = self._calculate_idf_probabilistic
        else:
            idf_map = {
                "standard": self._calculate_idf_standard,
                "probabilistic": self._calculate_idf_probabilistic,
                "smooth": self._calculate_idf_smooth,
                "max": self._calculate_idf_max
            }
            idf_fn = idf_map.get(idf_method, self._calculate_idf_standard)

        for t in q_terms:
            postings = self.index.get(t)
            if not postings:
                continue
            idf = idf_fn(t)

            for doc_id, tf_w in postings.items():
                # simple filtering
                if filters:
                    doc = self.documents.get(doc_id, {})
                    ok = True
                    for k, v in filters.items():
                        if doc.get(k) != v:
                            ok = False
                            break
                    if not ok:
                        continue

                if use_bm25:
                    k1 = bm25_params.get("k1", 1.5)
                    b = bm25_params.get("b", 0.75)
                    dl = max(1, self.doc_lengths.get(doc_id, 1))
                    tf = tf_w  # weighted term freq
                    tf_comp = (tf * (k1 + 1.0)) / (tf + k1 * (1 - b + b * (dl / (self.avg_doc_length or 1))))
                    scores[doc_id] += idf * tf_comp
                else:
                    scores[doc_id] += tf_w * idf

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def compare_idf_methods(self, query: str, top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        methods = ["standard", "probabilistic", "smooth", "max"]
        results = {}

        print(f"Comparing IDF methods for query: {query}")
        print("=" * 80)

        for method in methods:
            start_time = time.time()
            method_results = self.search(query, top_k=top_k, idf_method=method)
            query_time = time.time() - start_time

            results[method] = method_results

            print(f"{method.upper()} IDF (Time: {query_time:.3f}s):")
            for i, (doc_id, score) in enumerate(method_results, 1):
                doc = self.documents[doc_id]
                title = doc.get('title', 'No Title')[:50]
                print(f"  {i}. {title} (score: {score:.3f})")

        return results

    def analyze_term(self, term: str) -> Dict[str, Any]:
        term_lower = term.lower()
        doc_freq = len(self.index.get(term_lower, {}))

        analysis = {
            "term": term,
            "document_frequency": doc_freq,
            "total_documents": self.doc_count,
            "idf_values": {
                "standard": self._calculate_idf_standard(term_lower),
                "probabilistic": self._calculate_idf_probabilistic(term_lower),
                "smooth": self._calculate_idf_smooth(term_lower),
                "max": self._calculate_idf_max(term_lower),
            }
        }
        return analysis

    def get_index_stats(self) -> Dict[str, Any]:
        total_terms = sum(self.doc_lengths.values())
        unique_terms = len(self.index)

        df_counts = defaultdict(int)
        for term_docs in self.index.values():
            df = len(term_docs)
            df_counts[df] += 1

        return {
            "total_documents": self.doc_count,
            "total_terms": total_terms,
            "unique_terms": unique_terms,
            "average_document_length": self.avg_doc_length,
            "document_frequency_distribution": dict(sorted(df_counts.items())),
            "most_frequent_terms": self.get_most_frequent_terms(10)
        }

    def get_most_frequent_terms(self, top_n: int = 10) -> List[Tuple[str, int]]:
        term_freqs = []
        for term, doc_freqs in self.index.items():
            total_freq = sum(doc_freqs.values())
            term_freqs.append((term, total_freq, len(doc_freqs)))

        term_freqs.sort(key=lambda x: x[1], reverse=True)
        return [(term, int(freq), df) for term, freq, df in term_freqs[:top_n]]

    def save_to_file(self, filepath: str):
        index_data = {
            "index": self.index,
            "documents": self.documents,
            "doc_lengths": self.doc_lengths,
            "doc_count": self.doc_count,
            "avg_doc_length": self.avg_doc_length
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        self.index = index_data["index"]
        self.documents = index_data["documents"]
        self.doc_lengths = index_data["doc_lengths"]
        self.doc_count = index_data["doc_count"]
        self.avg_doc_length = index_data["avg_doc_length"]


class RTSearchEngine:
    def __init__(self, data_directory: str = "extracted_metadata"):
        self.data_directory = data_directory
        self.index = InvertedIndex()
        self.loaded = False

    def load_metadata(self, max_files: int = None) -> Dict[str, Any]:
        metadata = {}
        files_processed = 0

        if not os.path.exists(self.data_directory):
            print(f"Error: Data directory {self.data_directory} not found")
            return metadata

        for filename in os.listdir(self.data_directory):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.data_directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                doc = json.loads(line.strip())
                                # prefer a stable ID if present; fallback to file_path
                                doc_id = doc.get('file_path', f"doc_{len(metadata)}")
                                metadata[doc_id] = doc
                                files_processed += 1

                                if max_files and files_processed >= max_files:
                                    return metadata
                            except json.JSONDecodeError:
                                continue

        print(f"Loaded {len(metadata)} documents")
        return metadata

    def build_index(self, max_docs: int = None):
        print("Loading metadata...")
        metadata = self.load_metadata(max_docs)

        if not metadata:
            print("No metadata found! Please check your data directory.")
            return False

        print("Building inverted index...")
        for doc_id, doc in metadata.items():
            self.index.add_document(doc_id, doc)

        self.loaded = True

        stats = self.index.get_index_stats()
        print(f"Indexing complete!")
        print(f"  Documents: {stats['total_documents']}")
        print(f"  Unique terms: {stats['unique_terms']}")
        print(f"  Average document length: {stats['average_document_length']:.1f} terms")
        return True

    def search(self, query: str, top_k: int = 10,
               idf_method: str = "standard",
               use_bm25: bool = False) -> List[Dict[str, Any]]:
        if not self.loaded:
            raise ValueError("Index not built. Call build_index() first.")

        results = self.index.search(query, top_k=top_k, idf_method=idf_method, use_bm25=use_bm25)

        formatted_results = []
        for doc_id, score in results:
            doc_data = self.index.documents[doc_id].copy()
            doc_data['score'] = score
            doc_data['doc_id'] = doc_id
            formatted_results.append(doc_data)

        return formatted_results

    def compare_search_methods(self, query: str, top_k: int = 5):
        if not self.loaded:
            raise ValueError("Index not built. Call build_index() first.")

        print(f"Comparing ranking methods for: {query}")
        print("=" * 80)

        methods = [
            ("TF-IDF (Standard IDF)", "standard", False),
            ("TF-IDF (Probabilistic IDF)", "probabilistic", False),
            ("TF-IDF (Smoothed IDF)", "smooth", False),
            ("TF-IDF (Max IDF)", "max", False),
            ("BM25 (Standard)", "standard", True),
        ]

        all_results = {}

        for method_name, idf_method, use_bm25 in methods:
            start_time = time.time()
            results = self.search(query, top_k=top_k, idf_method=idf_method, use_bm25=use_bm25)
            query_time = time.time() - start_time

            all_results[method_name] = results

            print(f"{method_name} (Time: {query_time:.3f}s):")
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No Title')[:50]
                score = result.get('score', 0)
                page_type = result.get('page_type', 'unknown')
                print(f"  {i}. {title} ({page_type}) - Score: {score:.3f}")

    def analyze_query_terms(self, query: str):
        if not self.loaded:
            raise ValueError("Index not built. Call build_index() first.")

        terms = self.index._tokenize(query)

        print(f"Term analysis for query: {query}")
        print("=" * 60)

        for term in terms:
            analysis = self.index.analyze_term(term)
            print(f"Term: {term}")
            print(f"  Document Frequency: {analysis['document_frequency']}/{analysis['total_documents']}")
            print("  IDF Values:")
            for method, value in analysis['idf_values'].items():
                print(f"    {method}: {value:.4f}")

    def show_index_statistics(self):
        if not self.loaded:
            raise ValueError("Index not built. Call build_index() first.")

        stats = self.index.get_index_stats()

        print("=" * 60)
        print("INDEX STATISTICS")
        print("=" * 60)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Terms: {stats['total_terms']:,}")
        print(f"Unique Terms: {stats['unique_terms']:,}")
        print(f"Average Document Length: {stats['average_document_length']:.1f} terms")

        print(f"Most Frequent Terms:")
        for i, (term, freq, df) in enumerate(stats['most_frequent_terms'], 1):
            print(f"  {i}. {term}: {freq:,} total occurrences, {df} documents")

        print(f"Document Frequency Distribution:")
        for df, count in list(stats['document_frequency_distribution'].items())[:10]:
            print(f"  {df} documents: {count} terms")
        if len(stats['document_frequency_distribution']) > 10:
            print(f"  ... and {len(stats['document_frequency_distribution']) - 10} more frequency levels")

    def save_index(self, filepath: str = "search_index.json"):
        self.index.save_to_file(filepath)
        print(f"Index saved to: {filepath}")

    def load_index(self, filepath: str = "search_index.json"):
        if not os.path.exists(filepath):
            print(f"Index file not found: {filepath}")
            return False

        self.index.load_from_file(filepath)
        self.loaded = True
        print(f"Index loaded from: {filepath}")
        print(f"  Documents: {self.index.doc_count}")
        print(f"  Unique terms: {len(self.index.index)}")
        return True


class SearchMenu:
    def __init__(self):
        self.engine = RTSearchEngine()
        self.data_dir = "extracted_metadata"
        self.setup_complete = False

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        print("\n" + "=" * 70)
        print("           ROTTEN TOMATOES SEARCH ENGINE - IDF COMPARISON")
        print("=" * 70)

    def print_menu(self):
        print("\nMAIN MENU:")
        print("1. Build Search Index")
        print("2. Load Existing Index")
        print("3. Search with Specific IDF Method")
        print("4. Compare All IDF Methods")
        print("5. Analyze Query Terms")
        print("6. Show Index Statistics")
        print("7. Save Current Index")
        print("8. Change Data Directory")
        print("0. Exit")
        print("-" * 70)

    def wait_for_enter(self):
        input("\nPress Enter to continue...")

    def setup_engine(self):
        self.clear_screen()
        self.print_header()

        print("\nSETUP SEARCH ENGINE")
        print("1. Build new index from metadata")
        print("2. Load existing index from file")
        print("3. Back to main menu")

        choice = input("\nChoose option (1-3): ").strip()

        if choice == "1":
            max_docs = input("Max documents to index (Enter for all): ").strip()
            max_docs = int(max_docs) if max_docs.isdigit() else None

            if self.engine.build_index(max_docs):
                self.setup_complete = True
            else:
                print("Failed to build index")
            self.wait_for_enter()

        elif choice == "2":
            filename = input("Index filename [search_index.json]: ").strip()
            if not filename:
                filename = "search_index.json"

            if self.engine.load_index(filename):
                self.setup_complete = True
                print("Index loaded successfully!")
            else:
                print("Failed to load index")
            self.wait_for_enter()

    def search_with_method(self):
        if not self.setup_complete:
            print("Please setup the search engine first (Option 1 or 2)")
            self.wait_for_enter()
            return

        self.clear_screen()
        self.print_header()

        print("\nSEARCH WITH SPECIFIC IDF METHOD")
        query = input("Enter search query: ").strip()

        if not query:
            print("No query entered!")
            self.wait_for_enter()
            return

        print("\nAvailable IDF Methods:")
        print("1. Standard IDF")
        print("2. Probabilistic IDF")
        print("3. Smoothed IDF")
        print("4. Max IDF")
        print("5. BM25 Ranking")

        method_choice = input("\nChoose method (1-5): ").strip()

        method_map = {
            "1": ("standard", False, "Standard IDF"),
            "2": ("probabilistic", False, "Probabilistic IDF"),
            "3": ("smooth", False, "Smoothed IDF"),
            "4": ("max", False, "Max IDF"),
            "5": ("standard", True, "BM25 Ranking")
        }

        if method_choice not in method_map:
            print("Invalid choice!")
            self.wait_for_enter()
            return

        idf_method, use_bm25, method_name = method_map[method_choice]

        try:
            start_time = time.time()
            results = self.engine.search(query, top_k=5, idf_method=idf_method, use_bm25=use_bm25)
            query_time = time.time() - start_time

            self.display_results(results, query, method_name, query_time)

        except Exception as e:
            print(f"Search error: {e}")

        self.wait_for_enter()

    def compare_methods(self):
        if not self.setup_complete:
            print("Please setup the search engine first (Option 1 or 2)")
            self.wait_for_enter()
            return

        self.clear_screen()
        self.print_header()

        print("\nCOMPARE IDF METHODS")
        query = input("Enter search query: ").strip()

        if not query:
            print("No query entered!")
            self.wait_for_enter()
            return

        try:
            self.engine.compare_search_methods(query)
        except Exception as e:
            print(f"Error: {e}")

        self.wait_for_enter()

    def analyze_terms(self):
        if not self.setup_complete:
            print("Please setup the search engine first (Option 1 or 2)")
            self.wait_for_enter()
            return

        self.clear_screen()
        self.print_header()

        print("\nANALYZE QUERY TERMS")
        query = input("Enter query to analyze: ").strip()

        if not query:
            print("No query entered!")
            self.wait_for_enter()
            return

        try:
            self.engine.analyze_query_terms(query)
        except Exception as e:
            print(f"Error: {e}")

        self.wait_for_enter()

    def show_stats(self):
        if not self.setup_complete:
            print("Please setup the search engine first (Option 1 or 2)")
            self.wait_for_enter()
            return

        self.clear_screen()
        self.print_header()

        print("\nINDEX STATISTICS")
        try:
            self.engine.show_index_statistics()
        except Exception as e:
            print(f"Error: {e}")

        self.wait_for_enter()

    def save_index(self):
        if not self.setup_complete:
            print("Please setup the search engine first (Option 1 or 2)")
            self.wait_for_enter()
            return

        self.clear_screen()
        self.print_header()

        print("\nSAVE INDEX")
        filename = input("Filename [search_index.json]: ").strip()
        if not filename:
            filename = "search_index.json"

        try:
            self.engine.save_index(filename)
        except Exception as e:
            print(f"Error saving index: {e}")

        self.wait_for_enter()

    def change_data_dir(self):
        self.clear_screen()
        self.print_header()

        print(f"\nCURRENT DATA DIRECTORY: {self.data_dir}")
        new_dir = input("New data directory: ").strip()

        if new_dir:
            self.data_dir = new_dir
            self.engine = RTSearchEngine(self.data_dir)
            self.setup_complete = False
            print(f"Data directory changed to: {self.data_dir}")
        else:
            print("Directory unchanged.")

        self.wait_for_enter()

    def display_results(self, results: List[Dict[str, Any]], query: str, method: str, query_time: float):
        if not results:
            print("No results found.")
            return

        print(f"\n{method} - Found {len(results)} results for {query} in {query_time:.3f}s")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.get('title', 'No Title')}")
            print(f"   Type: {result.get('page_type', 'Unknown')}")
            print(f"   Score: {result.get('score', 0):.4f}")

            # URL or file path
            if result.get('url'):
                print(f"   URL: {result['url']}")
            elif result.get('file_path'):
                print(f"   File: {result['file_path']}")

            # metadata
            if result.get('tomatometer') is not None:
                print(f"   Tomatometer: {result['tomatometer']}%")
            if result.get('audience_score') is not None:
                print(f"   Audience Score: {result['audience_score']}%")
            if result.get('genres'):
                print(f"   Genres: {', '.join(result['genres'])}")
            if result.get('rating'):
                print(f"   Rating: {result['rating']}")
            if result.get('release_date'):
                print(f"   Release date: {result['release_date']}")
            if result.get('director'):
                print(f"   Director: {result['director']}")

            cast = result.get('cast')
            if cast:
                if isinstance(cast, list):
                    shown = [str(x) for x in cast[:6]]
                    extra = len(cast) - 6
                    cast_str = ", ".join(shown) + (f" +{extra} more" if extra > 0 else "")
                else:
                    cast_str = str(cast)
                print(f"   Cast: {cast_str}")

            if result.get('synopsis'):
                syn = result['synopsis']
                if len(syn) > 220:
                    syn = syn[:217] + "..."
                print(f"   Synopsis: {syn}")

            print("-" * 40)

    def run(self):
        while True:
            self.clear_screen()
            self.print_header()

            if self.setup_complete:
                stats = self.engine.index.get_index_stats()
                print(f"Index Ready - {stats['total_documents']} documents, {stats['unique_terms']} unique terms")
            else:
                print("Index Not Loaded - Please setup search engine first")

            self.print_menu()

            choice = input("Enter your choice (0-8): ").strip()

            if choice == "0":
                print("\nExiting ...")
                break
            elif choice == "1":
                self.setup_engine()
            elif choice == "2":
                self.setup_engine()
            elif choice == "3":
                self.search_with_method()
            elif choice == "4":
                self.compare_methods()
            elif choice == "5":
                self.analyze_terms()
            elif choice == "6":
                self.show_stats()
            elif choice == "7":
                self.save_index()
            elif choice == "8":
                self.change_data_dir()
            else:
                print("Invalid choice! Please try again.")
                self.wait_for_enter()


if __name__ == "__main__":
    menu = SearchMenu()
    menu.run()
