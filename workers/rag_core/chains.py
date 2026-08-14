""" 
RAG agent for explaining financial terms using Qdrant retrieval + Gemini. 
"""
import os
from functools import lru_cache
from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .embeddings import get_embeddings
from .prompts import EXPLAIN_TERM_PROMPT

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DICTIONARY_COLLECTION = "finance_dictionary"
LLM_MODEL = "gemini-2.5-flash"
RETRIEVER_TOP_K = 3


def _format_docs(docs: List[Document]) -> str:
    """Join retrieved chunks, prefixing each with its source term for clarity."""
    lines = []
    for doc in docs:
        term = (doc.metadata or {}).get("term", "")
        prefix = f"[{term}] " if term else ""
        lines.append(f"{prefix}{doc.page_content}")
    return "\n\n".join(lines)


class RAGAgent:
    """Explains a financial term using Qdrant retrieval + Gemini generation."""

    def __init__(self, prompt_template: str = EXPLAIN_TERM_PROMPT):
        client = QdrantClient(url=QDRANT_URL)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=DICTIONARY_COLLECTION,
            embedding=get_embeddings(),
        )
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": RETRIEVER_TOP_K}
        )

        prompt = PromptTemplate(
            input_variables=["context", "term", "language"],
            template=prompt_template,
        )
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

        self.chain = (
            {
                "context": itemgetter("term") | self.retriever | _format_docs,
                "term": itemgetter("term"),
                "language": itemgetter("language"),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, user_input: dict) -> str:
        return self.chain.invoke(user_input)


@lru_cache(maxsize=1)
def get_agent() -> RAGAgent:
    return RAGAgent() #Lazy singleton