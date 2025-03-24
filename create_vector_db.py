import os
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def load_environment():
    _ = load_dotenv(find_dotenv())
    print("Environment variables loaded")

def load_documents(paper_directory="docs"):
    pdf_files = [f for f in os.listdir(paper_directory) if f.endswith('.pdf')]
    loaders = []
    
    for pdf in pdf_files:
        # Extract paper name without extension
        paper_name = os.path.splitext(pdf)[0]
        
        # Try to parse metadata from filename
        # Assuming filename format might contain year like: paper_JOURNAL_YEAR.pdf
        parts = paper_name.split('_')
        year = next((part for part in parts if part.isdigit() and len(part) == 4), "Unknown")
        journal = parts[1] if len(parts) > 1 else "Unknown"
        
        loader = PyPDFLoader(os.path.join(paper_directory, pdf))
        
        # Extend the load function to add metadata
        original_load = loader.load
        
        def load_with_metadata():
            docs = original_load()
            # Add metadata to each page
            for doc in docs:
                doc.metadata["source"] = pdf
                if "source" in doc.metadata and isinstance(doc.metadata["source"], str):
                    doc.metadata["title"] = paper_name
                    doc.metadata["year"] = year
                    doc.metadata["journal"] = journal
                    doc.metadata["is_main_paper"] = True  # Flag to identify main papers
            return docs
        
        loader.load = load_with_metadata
        loaders.append(loader)
    
    # Load the documents
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    print(f"Loaded {len(docs)} pages from {len(loaders)} papers")
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_database(chunks, persist_directory="./research_db"):
    embeddings = OpenAIEmbeddings()
    
    vectordb = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Vector database created and saved to {persist_directory}")
    return vectordb

def load_existing_vector_database(persist_directory="./research_db"):
    import os
    
    # Check if database exists
    if not os.path.exists(persist_directory):
        print(f"No database found at {persist_directory}. Creating a new one...")
        docs = load_documents()
        chunks = split_documents(docs)
        return create_vector_database(chunks, persist_directory=persist_directory)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Load vector database
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print(f"Loaded existing vector database from {persist_directory}")
    return vectordb

if __name__ == "__main__":
    load_environment()
    
    # Ask if we should create a new database or load existing
    print("Research Publications Chatbot - Vector Database Setup")
    print("----------------------------------------------------")
    create_new = input("Create new vector database? (y/n): ").lower() == 'y'
    
    if create_new:
        # Ask for paper directory
        paper_dir = input("Enter the directory containing your papers [papers]: ") or "papers"
        
        # Load and process documents
        docs = load_documents(paper_directory=paper_dir)
        
        # Ask for chunk size and overlap
        chunk_size = int(input("Enter chunk size [1000]: ") or "1000")
        chunk_overlap = int(input("Enter chunk overlap [200]: ") or "200")
        
        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Ask for database directory
        db_dir = input("Enter the directory to save the vector database [./research_db]: ") or "./research_db"
        
        create_vector_database(chunks, persist_directory=db_dir)
        print(f"Vector database created successfully in {db_dir}")
    else:
        # Load existing database
        db_dir = input("Enter the directory of the existing vector database [./research_db]: ") or "./research_db"
        load_existing_vector_database(persist_directory=db_dir)
        print(f"Vector database loaded successfully from {db_dir}")