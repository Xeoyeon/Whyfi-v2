import os
import time
import hashlib
import uuid
import logging
import re
from urllib.parse import quote
from datetime import datetime, timezone
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from transformers import AutoTokenizer
from observability.metrics import IngestionMetrics, TimedEmbeddings

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "finance_dictionary"
BATCH_SIZE = int(os.getenv("DICTIONARY_BATCH_SIZE", "16"))
MAX_WORKERS = int(os.getenv("DICTIONARY_MAX_WORKERS", "4"))
UPLOAD_CHUNK_SIZE = int(os.getenv("DICTIONARY_UPLOAD_CHUNK_SIZE", "64"))
WIKI_REQUEST_TIMEOUT = float(os.getenv("WIKI_REQUEST_TIMEOUT", "15"))
SLOW_FETCH_SECONDS = float(os.getenv("DICTIONARY_SLOW_FETCH_SECONDS", "30"))
WAIT_POLL_SECONDS = float(os.getenv("DICTIONARY_WAIT_POLL_SECONDS", "5"))
SOURCE_NAME = "Wikipedia"
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_REST_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
WIKI_SOURCE_PAGES = [
    page.strip()
    for page in os.getenv(
        "WIKI_SOURCE_PAGES",
        "|".join([
            "Outline of finance",
            "Glossary of stock market terms",
            "Glossary of economics",
            "Outline of economics",
            "Outline of accounting",
            "Financial market",
            "Stock market",
            "Investment",
            "Money market",
            "Financial system",
        ])
    ).split("|")
    if page.strip()
]
WIKI_USER_AGENT = os.getenv(
    "WIKI_USER_AGENT",
    "whyfi-v2-dictionary-ingestion/1.0 (https://github.com/whyfi-v2)"
)

_original_request = requests.sessions.Session.request
_wiki_session = requests.Session()
_wiki_session.headers.update({
    "User-Agent": WIKI_USER_AGENT,
    "Accept": "application/json",
})
_embeddings = None
_tokenizer = None


def _request_with_timeout(self, method, url, **kwargs):
    kwargs.setdefault("timeout", WIKI_REQUEST_TIMEOUT)
    return _original_request(self, method, url, **kwargs)


requests.sessions.Session.request = _request_with_timeout

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model BAAI/bge-m3.")
        _embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return _embeddings


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        logger.info("Loading tokenizer BAAI/bge-m3.")
        _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3") #for token counting
    return _tokenizer


def _response_preview(response):
    text = response.text.replace("\n", " ").replace("\r", " ")
    return text[:300]


def _wiki_get_json(url, params=None):
    response = _wiki_session.get(url, params=params)
    content_type = response.headers.get("content-type", "")

    if response.status_code != 200:
        logger.error(
            "Wikipedia request failed: status=%s content_type=%s url=%s preview=%r",
            response.status_code,
            content_type,
            response.url,
            _response_preview(response),
        )
        response.raise_for_status()

    if "json" not in content_type.lower():
        logger.error(
            "Wikipedia returned a non-JSON response: status=%s content_type=%s url=%s preview=%r",
            response.status_code,
            content_type,
            response.url,
            _response_preview(response),
        )
        raise ValueError(f"Wikipedia returned non-JSON response: {content_type}")

    try:
        return response.json()
    except ValueError:
        logger.exception(
            "Failed to decode Wikipedia JSON: status=%s content_type=%s url=%s preview=%r",
            response.status_code,
            content_type,
            response.url,
            _response_preview(response),
        )
        raise


def fetch_glossary_links():
    raw_terms = []
    request_count = 0

    for source_page in WIKI_SOURCE_PAGES:
        logger.info(f"Fetching glossary links from Wikipedia source page: {source_page}")
        params = {
            "action": "query",
            "titles": source_page,
            "prop": "links",
            "plnamespace": "0",
            "pllimit": "max",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
        }

        while True:
            data = _wiki_get_json(WIKI_API_URL, params=params)
            request_count += 1
            pages = data.get("query", {}).get("pages", [])

            if isinstance(pages, dict):
                pages = pages.values()

            for page in pages:
                if page.get("missing"):
                    logger.warning(
                        "Wikipedia source page is missing and will be skipped: %s",
                        page.get("title") or source_page,
                    )
                    continue

                page_links = page.get("links", [])
                logger.info(
                    "Fetched %s raw links from source page '%s' in this request.",
                    len(page_links),
                    page.get("title") or source_page,
                )

                for link in page_links:
                    title = link.get("title") or link.get("*")
                    namespace = link.get("ns")

                    if not title:
                        continue
                    if namespace is not None and str(namespace) != "0":
                        continue

                    raw_terms.append(title)

            continuation = data.get("continue")
            if not continuation:
                break

            params.update(continuation)

    terms = []
    seen_keys = set()
    for title in raw_terms:
        key = term_key(title)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        terms.append(title)

    logger.info(
        f"Fetched {len(raw_terms)} raw glossary links from Wikipedia "
        f"across {request_count} request(s) and {len(WIKI_SOURCE_PAGES)} source page(s); "
        f"{len(terms)} unique article terms remain."
    )
    return terms


def normalize_term(term):
    term = re.sub(r"\s+", " ", term or "").strip()
    term = term.split("(")[0].strip()
    return term


def term_key(term):
    return normalize_term(term).casefold()


def normalize_chunks(chunks):
    normalized_chunks = []

    for raw_chunk in chunks:
        chunk = re.sub(r"\s+", " ", raw_chunk or "").strip()
        if not chunk:
            continue

        leading_punctuation = re.match(r"^([.!?]+)(?!\d)\s*(.*)$", chunk)
        if leading_punctuation:
            punctuation, remainder = leading_punctuation.groups()
            if normalized_chunks and not normalized_chunks[-1].endswith(tuple(".!?")):
                normalized_chunks[-1] = f"{normalized_chunks[-1]}{punctuation}"
            chunk = remainder.strip()

        if chunk:
            normalized_chunks.append(chunk)

    return normalized_chunks


def load_existing_terms(client, collection_name):
    existing_terms = set()
    try:
        client.get_collection(collection_name)
        offset = None
        
        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=5000,
                offset=offset,
                with_payload=True,
                with_vectors=False # without vectors to speed up the scroll
            )
            
            for record in records:
                payload = record.payload or {}
                record_metadata = payload.get('metadata') or payload
                term = record_metadata.get('term')
                key = record_metadata.get('term_key') or term_key(term)
                if key:
                    existing_terms.add(key)
                    
            if offset is None:
                break
            
        logger.info(f"Fetched {len(existing_terms)} existing terms from collection '{collection_name}'.")
        
    except Exception as e:
        logger.error(f"Collection '{collection_name}' does not exist or error occurred: {e}")
        
    return existing_terms


def _fetch_term_data(term, clean_term):
    try:
        started_at = time.monotonic()
        encoded_term = quote(term.replace(" ", "_"), safe="")
        page = _wiki_get_json(f"{WIKI_REST_SUMMARY_URL}/{encoded_term}")
        page_type = page.get("type")

        if page_type == "disambiguation":
            return {"term": clean_term, "status": "error", "error": "Disambiguation page"}

        summary = page.get("extract")
        if not summary:
            return {"term": clean_term, "status": "error", "error": "No summary found"}

        elapsed = time.monotonic() - started_at

        if elapsed >= SLOW_FETCH_SECONDS:
            logger.warning(f"Slow Wikipedia fetch for term '{clean_term}' finished in {elapsed:.1f}s.")
        
        return{
            "term": clean_term,
            "term_key": term_key(clean_term),
            "content": summary,
            "url": page.get("content_urls", {}).get("desktop", {}).get("page"),
            "status": "success",
            "original_term": term,
            "fetch_elapsed": elapsed,
            "slow_fetch": elapsed >= SLOW_FETCH_SECONDS,
        }
        
    except requests.exceptions.Timeout:
        return {"term": clean_term, "status": "error", "error": "Timeout"}
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        return {"term": clean_term, "status": "error", "error": f"HTTP {status_code}"}
    except ValueError:
        return {"term": clean_term, "status": "error", "error": "API Response Error"}
    except Exception as e:
        logger.exception(f"Unexpected Wikipedia fetch error for term '{clean_term}'.")
        return {"term": clean_term, "status": "error", "error": str(e)}
     

def chunk_and_upsert_batch(vector_store, batch_data, metrics=None):
    if not batch_data:
        logger.info("No new terms to process and upload.")
        return

    logger.info(f"[batch] Preparing {len(batch_data)} terms for chunking and upload.")
    documents, metadata, chunk_ids =[], [], []
    chunk_sources = []
    now = datetime.now(timezone.utc).isoformat()
    
    for item in batch_data:
        if not item.get('content'):
            continue
        
        clean_content = item['content'].strip()
        chunks = normalize_chunks(text_splitter.split_text(clean_content))
        
        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            chunk_sources.append((item, idx, chunk))
            
            chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{item['term_key']}_chunk_{idx}"))
            chunk_ids.append(chunk_uuid)

    if not documents:
        logger.info("No valid chunks found in this batch.")
        return

    logger.info(f"[batch] Tokenizing {len(documents)} chunks.")
    tokenizer = get_tokenizer()
    token_counts = [len(tokens) for tokens in tokenizer.batch_encode_plus(documents, add_special_tokens=True)["input_ids"]]
    for (item, _idx, chunk), token_count in zip(chunk_sources, token_counts):
        document_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item['term_key']))
        metadata.append({
            "term": item['term'],
            "term_key": item['term_key'],
            "original_term": item.get('original_term'),
            "document_id": document_id,
            "source": SOURCE_NAME,
            "link": item.get('url'),
            "type": "dictionary",
            "char_length": len(chunk),
            "token_count": token_count,
            "content_hash": hashlib.md5(chunk.encode()).hexdigest(),
            "updated_at": now
        })

    for start in range(0, len(documents), UPLOAD_CHUNK_SIZE):
        end = start + UPLOAD_CHUNK_SIZE
        logger.info(f"[batch] Uploading chunks {start + 1}-{min(end, len(documents))} of {len(documents)}.")
        upload_started_at = time.monotonic()
        embedding_before = metrics.embedding_seconds if metrics else 0.0
        vector_store.add_texts(
            documents[start:end],
            metadatas=metadata[start:end],
            ids=chunk_ids[start:end]
        )
        upload_elapsed = time.monotonic() - upload_started_at
        embedding_elapsed = (metrics.embedding_seconds - embedding_before) if metrics else 0.0
        logger.info(f"[batch] Upload chunk finished in {upload_elapsed:.1f}s.")

        if metrics:
            metrics.record_batch_upload_time(upload_elapsed)
            metrics.record_qdrant_upload_time(upload_elapsed - embedding_elapsed)

    if metrics:
        metrics.record_uploaded_chunks(len(documents))
    
    logger.info(f"[batch] Uploaded {len(documents)} chunks for {len(batch_data)} terms to Qdrant collection '{COLLECTION_NAME}'.")
  
  
def _log_slow_futures(in_flight):
    now = time.monotonic()
    slow_terms = []

    for info in in_flight.values():
        elapsed = now - info["started_at"]
        if elapsed < SLOW_FETCH_SECONDS or now - info["last_logged_at"] < SLOW_FETCH_SECONDS:
            continue
        info["last_logged_at"] = now
        slow_terms.append(f"{info['term']} ({elapsed:.1f}s)")

    if slow_terms:
        logger.warning(
            "Waiting on slow Wikipedia fetches: "
            + "; ".join(slow_terms[:10])
            + (" ..." if len(slow_terms) > 10 else "")
        )


def run_ingestion_pipeline(qdrant_client, existing_terms):
    metrics = IngestionMetrics()
    logger.info("Starting term collection from Wikipedia...")
    logger.info(
        "Runtime settings: "
        f"workers={MAX_WORKERS}, term_batch={BATCH_SIZE}, upload_chunk={UPLOAD_CHUNK_SIZE}, "
        f"wiki_timeout={WIKI_REQUEST_TIMEOUT}s, source_pages={WIKI_SOURCE_PAGES}"
    )
    vector_store = None
    
    try:
        all_terms = fetch_glossary_links()
    except Exception as e:
        logger.exception(f"Error fetching Wikipedia source pages: {e}")
        metrics.log_summary()
        return
    
    invalid_keywords = [
        "List of", "Glossary", "Index of", "Outline of", "Timeline of", "History of", 
        "Category:", "Portal:", "Help:", "Special:", "disambiguation"
    ]
    
    new_terms = []
    seen_terms = set(existing_terms)
    for term in all_terms:
        # filter out irrelevant terms based on keywords and length
        if len(term) > 100 or len(term) < 3:
            continue
        
        term_lower = term.lower()
        if any(keyword.lower() in term_lower for keyword in invalid_keywords):
            continue
        
        clean_term = normalize_term(term)
        key = clean_term.casefold()
        
        if not key or key in seen_terms:
            continue
        
        seen_terms.add(key)
        new_terms.append((term, clean_term))
    
    logger.info(f"Number of new terms to collect: {len(new_terms)}")
    if not new_terms:
        logger.info("No new terms to fetch.")
        metrics.log_summary()
        return
    
    # Use ThreadPoolExecutor to fetch term data concurrently
    batch_data=[]

    def submit_next_term(executor, term_iter, in_flight):
        try:
            term, clean_term = next(term_iter)
        except StopIteration:
            return False

        future = executor.submit(_fetch_term_data, term, clean_term)
        in_flight[future] = {
            "term": clean_term,
            "started_at": time.monotonic(),
            "last_logged_at": time.monotonic(),
        }
        return True

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        term_iter = iter(new_terms)
        in_flight = {}

        for _ in range(MAX_WORKERS):
            if not submit_next_term(executor, term_iter, in_flight):
                break

        with tqdm(total=len(new_terms), desc="data scraping", unit="term") as progress:
            while in_flight:
                done, _pending = wait(
                    in_flight.keys(),
                    timeout=WAIT_POLL_SECONDS,
                    return_when=FIRST_COMPLETED
                )

                if not done:
                    _log_slow_futures(in_flight)
                    continue

                for future in done:
                    future_info = in_flight.pop(future)
                    progress.update(1)

                    try:
                        result = future.result()
                    except Exception:
                        logger.exception(f"Unexpected fetch error for term '{future_info['term']}'.")
                        metrics.record_failure("Unexpected fetch error")
                        submit_next_term(executor, term_iter, in_flight)
                        continue
            
                    if result["status"] == "success":
                        if result["term_key"] in existing_terms:
                            submit_next_term(executor, term_iter, in_flight)
                            continue
                        batch_data.append(result)
                        existing_terms.add(result["term_key"])
                        metrics.record_success(slow_fetch=result.get("slow_fetch", False))
                    else:
                        error_reason = result.get("error")
                        metrics.record_failure(error_reason)
                        logger.warning(f"Failed to fetch data for term '{result['term']}': {error_reason}")

                    if len(batch_data) >= BATCH_SIZE:
                        try:
                            if vector_store is None:
                                vector_store = QdrantVectorStore(
                                    client=qdrant_client,
                                    collection_name=COLLECTION_NAME,
                                    embedding=TimedEmbeddings(get_embeddings(), metrics)
                                )
                            chunk_and_upsert_batch(vector_store, batch_data, metrics)
                        except MemoryError:
                            logger.exception(
                                "Out of memory while processing a batch. "
                                "Try lowering DICTIONARY_BATCH_SIZE or DICTIONARY_UPLOAD_CHUNK_SIZE."
                            )
                            metrics.log_summary()
                            raise
                        except Exception:
                            logger.exception("Batch upload failed.")
                            metrics.log_summary()
                            raise
                        finally:
                            batch_data.clear()

                    submit_next_term(executor, term_iter, in_flight)

    if batch_data:
        try:
            if vector_store is None:
                vector_store = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=COLLECTION_NAME,
                    embedding=TimedEmbeddings(get_embeddings(), metrics)
                )
            chunk_and_upsert_batch(vector_store, batch_data, metrics)
        except MemoryError:
            logger.exception(
                "Out of memory while processing the final batch. "
                "Try lowering DICTIONARY_BATCH_SIZE or DICTIONARY_UPLOAD_CHUNK_SIZE."
            )
            metrics.log_summary()
            raise
        except Exception:
            logger.exception("Final batch upload failed.")
            metrics.log_summary()
            raise
        batch_data.clear()
    
    logger.info(f"Successfully collected data for {metrics.success_terms} terms.")
    metrics.log_summary()


    
if __name__ == "__main__":
    qdrant_client = QdrantClient(url=QDRANT_URL)
    existing_terms = load_existing_terms(qdrant_client, COLLECTION_NAME)
    run_ingestion_pipeline(qdrant_client, existing_terms)
    logger.info("Term collection and upload process completed.")
