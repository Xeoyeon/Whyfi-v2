import os
import requests
import time
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "finance_dictionary"

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
# Test data for development purposes. In production, this function should scrape real data from Investopedia or another reliable source.
# def collect_finance_terms():
#     sample_data = [
#         {"term": "Asset (자산)", "content": "An asset is a resource with economic value that an individual, corporation, or country owns or controls with the expectation that it will provide future benefit."},
#         {"term": "Liability (부채)", "content": "A liability is a company's legal financial debts or obligations that arise during the course of business operations."},
#         {"term": "Equity (자본/자기자본)", "content": "Equity represents the shareholders' stake in the company, identified on a company's balance sheet; calculated as total assets minus total liabilities."},
#         {"term": "Interest Rate (금리)", "content": "An interest rate is the amount charged, expressed as a percentage of principal, by a lender to a borrower for the use of assets."},
#         {"term": "Inflation (인플레이션)", "content": "Inflation is the decline of purchasing power of a given currency over time, often measured by the rise in the general level of prices for goods and services."},
#         {"term": "Bull Market (강세장)", "content": "A bull market is the condition of a financial market in which prices are rising or are expected to rise."},
#         {"term": "Bear Market (약세장)", "content": "A bear market is when a market experiences prolonged price declines, typically falling 20% or more from recent highs."},
#         {"term": "Liquidity (유동성)", "content": "Liquidity refers to the efficiency or ease with which an asset or security can be converted into ready cash without affecting its market price."},
#         {"term": "Dividend (배당금)", "content": "A dividend is the distribution of some of a company's earnings to a class of its shareholders, as determined by the company's board of directors."},
#         {"term": "Volatility (변동성)", "content": "Volatility is a statistical measure of the dispersion of returns for a given security or market index. Higher volatility means higher risk."},
#         {"term": "ETF (상장지수펀드)", "content": "An exchange-traded fund (ETF) is a type of pooled investment security that operates much like a mutual fund but trades like a regular stock on an exchange."},
#         {"term": "GDP (국내총생산)", "content": "Gross Domestic Product (GDP) is the total monetary or market value of all the finished goods and services produced within a country's borders in a specific time period."},
#         {"term": "Market Capitalization (시가총액)", "content": "Market capitalization refers to the total dollar market value of a company's outstanding shares of stock, calculated by multiplying total shares by the current market price."},
#         {"term": "Quantitative Easing (양적완화)", "content": "Quantitative easing (QE) is a form of monetary policy in which a central bank purchases longer-term securities from the open market to increase the money supply and encourage lending."},
#         {"term": "Yield Curve (수익률 곡선)", "content": "A yield curve is a line that plots yields (interest rates) of bonds having equal credit quality but differing maturity dates. An inverted yield curve is often seen as an indicator of an impending economic recession."}
#     ]
#     return sample_data

def collect_finance_terms():
    print("Collecting finance terms from Investopedia...")
    
    url ="https://www.investopedia.com/financial-term-dictionary-4769738"
    
    # 봇(Bot)으로 인식되어 차단당하는 것을 방지하기 위한 User-Agent 헤더
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/", # 구글에서 타고 들어온 척하기
        "Connection": "keep-alive",
    }
    collected_data = []
    try :
        response = requests.get(url, headers=headers)
        response.raise_for_status() #200 OK가 아니면 에러 발생
    except Exception as e:
        print(f"main page 접근 실패: {e}")
        return collected_data
    
    soup = BeautifulSoup(response.text, 'lxml')
    # Investopedia 사전 페이지에서 용어 링크들 추출 (예시용 CSS 셀렉터)
    # 실제 사이트 구조 변경 시 select 내부의 클래스명을 수정해야 할 수 있습니다.
    term_links = soup.select("a.dictionary-top300-list__list-item, .dictionary-top300-list a")
    
    if not term_links:
        print("No term links found. Please check the CSS selector and website structure.")
        return collected_data
    print(f"Found {len(term_links)} term links. Collecting data...")
    
    for link in term_links[:5]:
        term_name = link.text.strip()
        term_url = link['href']
        
        if not term_url or not term_url.startswith("http"):
            continue
        try:
            # 개별 용어 페이지 접속
            term_response = requests.get(term_url, headers=headers)
            term_soup = BeautifulSoup(term_response.text, 'lxml')
            
            # 본문 문단(<p>) 추출 로직
            # 보통 class에 'mntl-sc-page' 나 'article-body'가 포함되어 있습니다.
            paragraphs = term_soup.select("#mntl-sc-page_1-0 p, .mntl-sc-page p, .article-body p")
            
            if paragraphs:
                content = " ".join(p.text.strip() for p in paragraphs[:3] if p.text.strip()) #첫 3개 문단 정도만 합쳐서 핵심 의미로 사용
                collected_data.append({
                    "term": term_name,
                    "content": content
                })
                print(f"Collected term: {term_name}")
            else:
                print(f"No content found for term: {term_name}")
                
            time.sleep(1)
            
        except Exception as e:
            print(f"Error collecting term {term_name}: {e}")
            
    return collected_data
        

def preprocess_and_upload(raw_data):
    if not raw_data:
        print("No data to preprocess and upload.")
        return
    
    print("Preprocessing data and uploading to Qdrant...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    
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
                "source": "Investopedia",
                "type": "dictionary"
            })
    print(f"Generated {len(documents)} chunks from {len(raw_data)} terms. Uploading to Qdrant...")
    
    qdrant_client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    vector_store.add_texts(documents, metadatas=metadata)
    print("<Finance_dictionary> Data uploaded to Qdrant successfully.")
    
if __name__ == "__main__":
    data = collect_finance_terms()
    preprocess_and_upload(data)