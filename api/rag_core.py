import os
from typing import List
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

prompt_template = """
You are a helpful and friendly chatbot assistant. Your task is to explain the given financial term simply and clearly in everyday language.
All your responses MUST be written in {language}.

User Input Term: 
{term}

Context:
{word_context}
{book_context}

<hr>
<h3>💡<b>{term}</b></h3>
[Explain the meaning of the term concisely and clearly in about 3-4 sentences in {language}] 

<h3>💚<b>Examples</b></h3>  
[Provide a simple, real-life example to help understand the term in {language}]

<h3>🔍<b>Related Words</b></h3>
<ol>
    <li> [Related Word 1]</li>  
    <li> [Related Word 2]</li>  
    <li> [Related Word 3]</li>
</ol>
<hr>
"""

class ChromaDB:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    def get_collection(self, collection_name: str):
        return Chroma(
            collection_name=collection_name, 
            persist_directory="../chroma_index", 
            embedding_function=self.embedding_model
        )

class RAGAgent:
    def __init__(self, prompt_template):
        db = ChromaDB()
        self.word_retriever = db.get_collection("words700").as_retriever(search_kwargs={"k": 3})
        self.book_retriever = db.get_collection("stock_book").as_retriever(search_kwargs={"k": 2})
        
        prompt = PromptTemplate(
            input_variables=["word_context", "book_context", "term", "language"], 
            template=prompt_template
        )
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        self.chain = (
            {
                "word_context": itemgetter("term") | self.word_retriever | self.format_retriever_output,
                "book_context": itemgetter("term") | self.book_retriever | self.format_retriever_output,
                "term": itemgetter("term"),
                "language": itemgetter("language")
            }
            | prompt
            | llm 
            | StrOutputParser()
        )

    def format_retriever_output(self, docs: List[Document]) -> str:
        return "\n".join([doc.page_content for doc in docs])
        
    def invoke(self, user_input: dict) -> str:
        return self.chain.invoke(user_input)

ce_agent = RAGAgent(prompt_template=prompt_template)