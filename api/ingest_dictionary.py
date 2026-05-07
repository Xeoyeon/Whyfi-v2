import os
import time
import wikipedia
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "finance_dictionary"

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

def get_existing_terms(client, collection_name):
    existing_terms = set()
    try:
        client.get_collection(collection_name)

        records, _ = client.scroll(
            collection_name=collection_name,
            limit=10000, 
            with_payload=True,
            with_vectors=False # without vectors to speed up the scroll
        )
        
        for record in records:
            metadata = record.payload.get('metadata', {})
            term = metadata.get('term')
            if term:
                existing_terms.add(term)
        print(f"Found {len(existing_terms)} existing terms in the collection.")
        
    except Exception as e:
        print(f"Collection '{collection_name}' does not exist or error occurred: {e}")
        
    return existing_terms


def collect_finance_terms(existing_terms):
    print("Collecting finance terms in Wikipedia's 'Glossary of finance'...")
    wikipedia.set_lang("en")
    
    try :
        glossary_page = wikipedia.page("Glossary of finance")
        all_terms= glossary_page.links
    except Exception as e:
        print(f"Error fetching glossary page: {e}")
        return []
    
    collected_data = []
    invalid_keywords = ["List of", "Glossary", "Index of", "Outline of", "Timeline of", "History of", "Category:", "Portal:", "Help:", "Special:", "Portal:","disambiguation"]
    #target_terms = all_terms[:30] #for test
    
    for term in tqdm(all_terms, desc="Processing terms", unit="term"):
        # filter out irrelevant terms based on keywords and length
        if any(keyword in term for keyword in invalid_keywords):
            continue
        if len(term) > 100 or len(term) < 3:
            continue
        
        clean_term = term.split('(')[0].strip()
        
        if clean_term in existing_terms:
            print(f"Skipping existing term: {clean_term}")
            continue
        
        try:
            summary = wikipedia.summary(term, sentences=5)
            url_term=term.replace(" ", "_")
            wiki_link=f"https://en.wikipedia.org/wiki/{url_term}"
            collected_data.append({
                "term": clean_term,
                "content": summary,
                "url": wiki_link
            })
            print(f"✅Collected data for term: {term}")
            time.sleep(0.5)
        except ValueError as e:
            print(f"API Server response error for term '{term}': {e}")
            time.sleep(2)   
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error for term '{term}': {e}")
        except wikipedia.exceptions.PageError as e:
            print(f"Page not found for term '{term}': {e}")
        except Exception as e:
            print(f"Error collecting data for term '{term}': {e}")
            
    return collected_data
        

def preprocess_and_upload(raw_data):
    if not raw_data:
        print("No data to preprocess and upload.")
        return
    
    print(f"Preprocessing {len(raw_data)} new terms and uploading to Qdrant...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""] 
    )
    
    documents=[]
    metadata=[]
    
    for item in raw_data:
        if not item.get('content'):
            continue
        clean_content = item['content'].strip()
        chunks = text_splitter.split_text(clean_content)
        
        for chunk in chunks:
            documents.append(chunk)
            metadata.append({
                "term": item['term'],
                "source": "Wikipedia",
                "link":item.get('url'),
                "type": "dictionary"
            })
    
    qdrant_client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    vector_store.add_texts(documents, metadatas=metadata)
    print("<Finance_dictionary> Data uploaded to Qdrant successfully.")
    
if __name__ == "__main__":
    client = QdrantClient(url=QDRANT_URL)
    existing_terms = get_existing_terms(client, COLLECTION_NAME)
    data = collect_finance_terms(existing_terms)
    preprocess_and_upload(data)