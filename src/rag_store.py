import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import json

load_dotenv()

def get_vector_store():
    """Initializes and returns the Chroma vector store with engagement strategies."""
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    client = chromadb.PersistentClient(path="./chroma_db")
    
    vector_store = Chroma(
        client=client,
        collection_name="retention_strategies",
        embedding_function=embeddings,
    )
    
    return vector_store

def seed_database():
    """Populates the database dynamically from a JSON knowledge base."""
    vector_store = get_vector_store()
    
    if vector_store._collection.count() > 0:
        print(f"Database already seeded with {vector_store._collection.count()} strategies.")
        return

    try:
        with open('data/retention_strategies.json', 'r') as file:
            raw_strategies = json.load(file)
    except FileNotFoundError:
        print("Error: data/retention_strategies.json not found. Please create it.")
        return

    documents = []
    for strat in raw_strategies:
        doc = Document(
            page_content=strat["strategy"],
            metadata={
                "category": strat["category"], 
                "target_metric": strat["trigger_metric"],
                "id": strat["id"]
            }
        )
        documents.append(doc)
    
    vector_store.add_documents(documents)
    print(f"Successfully embedded {len(documents)} retention strategies into ChromaDB.")

def retrieve_strategies(query_text: str, k: int = 2):
    """Retrieves the top k most relevant strategies for a given player scenario."""
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query_text, k=k)
    return results

if __name__ == "__main__":
    seed_database()