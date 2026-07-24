import os
import chromadb
import pdfplumber
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
DB_FOLDER = "./rag_database"
COLLECTION_NAME = "trading_playbooks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_database():
    """
    Connects to the local ChromaDB database and explicitly loads 
    the all-MiniLM-L6-v2 embedding model.
    """
    # 1. Load your chosen embedding model
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    
    # 2. Create or connect to the local folder database
    client = chromadb.PersistentClient(path=DB_FOLDER)
    
    # 3. Get or create the table (collection) where we store PDFs
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, 
        embedding_function=embed_fn
    )
    
    return collection

# ==========================================
# 2. PDF EXTRACTION & CHUNKING
# ==========================================
def extract_text_from_pdf(pdf_path):
    """Reads a PDF file and returns all its text as a single string."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=200, overlap=50):
    """
    Splits a massive wall of text into smaller, overlapping chunks (by word count).
    Overlap ensures sentences cut in half don't lose their context.
    """
    words = text.split()
    chunks = []
    
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        # Grab a slice of words and join them back into a string
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        
    return chunks

# ==========================================
# 3. BUILDING THE KNOWLEDGE BASE (Run Once)
# ==========================================
def build_knowledge_base(pdf_paths):
    """
    Reads a list of PDFs, chunks them, vectorizes them, and saves them to DB.
    You only need to run this function once (or when PDFs update).
    """
    collection = get_database()
    
    for path in pdf_paths:
        print(f"Processing: {path}...")
        
        # 1. Extract and chunk
        raw_text = extract_text_from_pdf(path)
        chunks = chunk_text(raw_text, chunk_size=200, overlap=50)
        
        # 2. Prepare data for the database
        documents = chunks
        metadatas = [{"source": path} for _ in chunks]
        ids = [f"{os.path.basename(path)}_chunk_{i}" for i in range(len(chunks))]
        
        # 3. Save to database (ChromaDB vectorizes it automatically here!)
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully saved {len(chunks)} chunks from {path}")

# ==========================================
# 4. THE AI RETRIEVAL TOOL (Agent uses this)
# ==========================================
def retrieve_strategy_for_agent(query, top_results=3):
    """
    The AI Agent will use this tool. 
    It takes a market regime name or a query, searches the DB, 
    and returns a formatted string of instructions.
    """
    collection = get_database()
    
    # Search the database. It automatically embeds the query string!
    search_results = collection.query(
        query_texts=[query],
        n_results=top_results
    )
    
    # Extract the matched text chunks
    retrieved_texts = search_results['documents'][0]
    sources = search_results['metadatas'][0]
    
    # Format the text neatly so the AI LLM can read and understand it
    output = f"### RAG Retrieval Results for: '{query}' ###\n\n"
    
    for i in range(len(retrieved_texts)):
        source_file = sources[i]['source']
        text_content = retrieved_texts[i]
        
        output += f"--- Source: {source_file} ---\n"
        output += f"{text_content}\n\n"
        
    return output

# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Define your PDFs
    my_pdfs = ["market_regime_playbook.pdf", "mkt_cycle.pdf"]
    
    # 2. BUILD THE DATABASE: This will actively run, vectorize, and save the PDFs to ChromaDB
    build_knowledge_base(my_pdfs)
    print("\n--- Database Build Complete ---\n")
    
    # 3. SINGLE LINE INVOCATION: Ask any query and fetch the answer immediately
    print(retrieve_strategy_for_agent("What is the strategy for an Established Bull Trend?"))