import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# --- Import the new local embeddings library ---
from langchain_community.embeddings import HuggingFaceEmbeddings

# Note: Caching imports are no longer strictly necessary for the main script flow 
# as we process all docs at once, but they are kept for the helper function.
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# Set up environment variables
# --- Import from the NEW, correct package ---
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def embed_and_save_documents():
    # --- Local Embedding Model Setup ---
    print("Initializing local embedding model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    # This line of code does not change, only the import above
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("✅ Local embedding model loaded.")

    loader = PyPDFDirectoryLoader("./LEGAL-DATA")
    print("Loader initialised")
    docs = loader.load()
    print("Loading the docs")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    print(f"Splitting the docs into {len(final_documents)} chunks.")
    
    for doc in final_documents:
        if 'source' in doc.metadata:
            doc.metadata['source'] = os.path.basename(doc.metadata['source'])
    
    if not final_documents:
        print("No documents found to process.")
        return

    print("Creating vector store from all documents...")
    vectors = FAISS.from_documents(final_documents, embeddings)
    
    print("All documents have been processed.")
    vectors.save_local("my_vector_store")
    print("Vector store saved successfully to 'my_vector_store'.")

# This helper function is also updated for consistency.
def add_text_to_vector_store(text, vector_store_path="my_vector_store"):
    # Initialize the same local model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    
    db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    db.add_texts([text])
    db.save_local(vector_store_path)

# --- How to run the main function ---
# To create your vector store, uncomment the following lines and run `python ingestion.py` in your terminal.
if __name__ == '__main__':
    embed_and_save_documents()