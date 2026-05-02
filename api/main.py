import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import fetch_naver_news, fetch_google_trends, fetch_popular_keywords
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from rag_core import ce_agent

load_dotenv()

app = FastAPI(title="Whyfi Enterprise API V2")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/explain")
def explain_term(
    term: str = Query(..., description="Financial term to explain"),
    lang: str = Query("en", description="Language selection (en or ko)") #default: English
):
    if not term:
        return {"error": "Please enter a financial term."}
    
    target_language = "English" if lang == "en" else "Korean"
    
    # 1. RAG inference
    explanation = ce_agent.invoke({"term": term, "language": target_language})
    
    # 2. Naver News Crawler
    news_items = fetch_naver_news(term)
    
    # 3. Google Trends
    trend_summary = ""
    try:
        trend = fetch_google_trends(term)
        if not trend.empty:
            avg_trend = round(trend['Trend Score'].mean(), 2)
            peak_trend = trend.loc[trend['Trend Score'].idxmax()]
            lowest_trend = trend.loc[trend['Trend Score'].idxmin()]
            
            trend_summary = f"Average Interest: {avg_trend}\n"
            trend_summary += f"Peak Interest: {int(peak_trend['Trend Score'])} ({peak_trend['Date'].strftime('%Y-%m-%d')})\n"
            trend_summary += f"Lowest Interest: {int(lowest_trend['Trend Score'])} ({lowest_trend['Date'].strftime('%Y-%m-%d')})"
    except Exception as e:
        print(f"Error fetching Google Trends data: {e}")
    
    return {
        "explanation": explanation,
        "news": news_items,
        "trend": trend_summary,
    }

@app.get("/keywords")
def get_keywords():
    keywords, date = fetch_popular_keywords()
        
    return {
        "keywords": keywords,
        "date": date,
    }