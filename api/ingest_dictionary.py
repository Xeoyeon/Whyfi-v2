import os
import time
import hashlib
import uuid
import logging
from datetime import datetime, timezone
import wikipedia
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "finance_dictionary"
BATCH_SIZE = 20

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3") #for token counting


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
                term = record.payload.get('metadata', {}).get('term')
                if term:
                    existing_terms.add(term)
                    
            if offset is None:
                break
            
        logger.info(f"Fetched {len(existing_terms)} existing terms from collection '{collection_name}'.")
        
    except Exception as e:
        logger.error(f"Collection '{collection_name}' does not exist or error occurred: {e}")
        
    return existing_terms


def _fetch_term_data(term, clean_term):
    try:
        summary = wikipedia.summary(term, sentences=5)
        url_term=term.replace(" ", "_")
        wiki_link=f"https://en.wikipedia.org/wiki/{url_term}" 
        time.sleep(0.5)
        
        return{
            "term": clean_term,
            "content": summary,
            "url": wiki_link,
            "status": "success",
            "original_term": term
        }
        
    except ValueError as e:
        time.sleep(2)
        return {"term": term, "status": "error", "error": "API Response Error"}
    except wikipedia.exceptions.PageError:
        return {"term": term, "status": "error", "error": "Page Not Found"}
    except Exception as e:
        return {"term": term, "status": "error", "error": str(e)}
     

def chunk_and_upsert_batch(qdrant_client, batch_data):
    if not batch_data:
        logger.info("No new terms to process and upload.")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""] 
    )
    
    documents, metadata, chunk_ids =[], [], []
    
    for item in batch_data:
        if not item.get('content'):
            continue
        
        clean_content = item['content'].strip()
        chunks = text_splitter.split_text(clean_content)
        document_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item['term']))
        
        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            
            metadata.append({
                "term": item['term'],
                "document_id": document_id,
                "source": "Wikipedia",
                "link":item.get('url'),
                "type": "dictionary",
                "char_length": len(chunk),
                "token_count": len(tokenizer.encode(chunk)),
                "content_hash": hashlib.md5(chunk.encode()).hexdigest(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            
            chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{item['term']}_chunk_{idx}"))
            chunk_ids.append(chunk_uuid)
            
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    vector_store.add_texts(documents, metadatas=metadata, ids=chunk_ids)
    
    logger.info(f"[batch] Uploaded {len(documents)} chunks for {len(batch_data)} terms to Qdrant collection '{COLLECTION_NAME}'.")
  
  
def run_ingestion_pipeline(qdrant_client,existing_terms):
    logger.info("Starting term collection from Wikipedia...")
    wikipedia.set_lang("en")
    
    try :
        all_terms= wikipedia.page("Glossary of finance").links
    except Exception as e:
        logger.error(f"Error fetching 'Glossary of finance' page: {e}")
        return
    
    invalid_keywords = [
        "List of", "Glossary", "Index of", "Outline of", "Timeline of", "History of", 
        "Category:", "Portal:", "Help:", "Special:", "disambiguation"
    ]
    
    target_terms = []
    for term in all_terms:
        # filter out irrelevant terms based on keywords and length
        if any(keyword in term for keyword in invalid_keywords):
            continue
        if len(term) > 100 or len(term) < 3:
            continue
        
        clean_term = term.split('(')[0].strip()
        
        if clean_term in existing_terms:
            continue
        target_terms.append((term, clean_term))
    
    logger.info(f"Number of new terms to collect: {len(target_terms)}")
    
    # Use ThreadPoolExecutor to fetch term data concurrently
    batch_data=[]
    total_success_count=0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_term_data, t[0], t[1]): t for t in target_terms}
        for future in tqdm(as_completed(futures), total=len(futures), desc="data scraping", unit="term"):
            result = future.result()
            
            if result["status"] == "success":
                batch_data.append(result)
                total_success_count += 1
            else:
                logger.warning(f"Failed to fetch data for term '{result['term']}': {result.get('error')}")
            
            if len(batch_data) >= BATCH_SIZE:
                chunk_and_upsert_batch(qdrant_client, batch_data)
                batch_data.clear()
    if batch_data:
        chunk_and_upsert_batch(qdrant_client, batch_data)
        batch_data.clear()
    
    logger.info(f"Successfully collected data for {total_success_count} terms.")


    
if __name__ == "__main__":
    qdrant_client = QdrantClient(url=QDRANT_URL)
    existing_terms = load_existing_terms(qdrant_client, COLLECTION_NAME)
    run_ingestion_pipeline(existing_terms)
    logger.info("Term collection and upload process completed.")
    