import re
import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use absolute paths for PDFs
PDF_FILES = [
    os.path.join(BASE_DIR, "1.pdf"),
    os.path.join(BASE_DIR, "2.pdf"),
    os.path.join(BASE_DIR, "3.pdf"),
    os.path.join(BASE_DIR, "4.pdf")
]
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "quantum_index")


def create_vector_store(pdf_files=PDF_FILES):
    """Create vector store from PDF files"""
    try:
        all_docs = []
        
        # Load all PDFs
        for pdf_path in pdf_files:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                continue
            
            logger.info(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {pdf_path}")
        
        if not all_docs:
            raise ValueError("No documents loaded from PDFs")
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(all_docs)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Create embeddings and vector store
        logger.info("Creating embeddings...")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Save vector store
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        vector_store.save_local(VECTOR_STORE_DIR)
        logger.info(f"Vector store saved to {VECTOR_STORE_DIR}")
        
        return vector_store
    
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise


def format_latex(text: str) -> str:
    """
    Detect formulas and wrap them in LaTeX delimiters.
    Preserves regular text while formatting mathematical expressions.
    """
    lines = text.split("\n")
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if line contains mathematical notation
        has_math = any(char in line for char in ["=", "^", "_", "∫", "∑", "∂", "∆", "√"])
        has_greek = bool(re.search(r'\\[a-zA-Z]+', line))  # LaTeX commands like \alpha
        
        if has_math or has_greek:
            # Wrap in display math if it's a standalone equation
            if line.count("=") >= 1 and len(line) < 100:
                formatted_lines.append(f"$${line}$$")
            else:
                # Inline math for shorter expressions
                formatted_lines.append(f"${line}$")
        else:
            formatted_lines.append(line)
    
    return "\n".join(formatted_lines)


def retrieve_answer(query: str, k: int = 3) -> str:
    """
    Retrieve relevant context from vector store based on query.
    
    Args:
        query: User's question
        k: Number of relevant chunks to retrieve
    
    Returns:
        Formatted context string
    """
    try:
        # Check if vector store exists
        if not os.path.exists(VECTOR_STORE_DIR):
            logger.warning(f"Vector store not found at {VECTOR_STORE_DIR}. Creating new one...")
            create_vector_store()
        
        # Load embeddings and vector store
        logger.info(f"Loading vector store for query: {query}")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(
            VECTOR_STORE_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Perform similarity search
        results = vector_store.similarity_search(query, k=k)
        
        if not results:
            logger.warning("No results found for query")
            return "No relevant information found in the documents."
        
        logger.info(f"Found {len(results)} relevant chunks")
        
        # Format results
        formatted_chunks = []
        for i, result in enumerate(results, 1):
            content = result.page_content.strip()
            formatted_content = format_latex(content)
            
            # Add source information if available
            source = result.metadata.get('source', 'Unknown')
            page = result.metadata.get('page', 'N/A')
            
            chunk_text = f"**Source {i}** (Page {page}):\n{formatted_content}"
            formatted_chunks.append(chunk_text)
        
        return "\n\n---\n\n".join(formatted_chunks)
    
    except Exception as e:
        error_msg = f"Error retrieving information: {str(e)}"
        logger.error(error_msg)
        return error_msg


# Check if vector store files exist
def vector_store_exists():
    """Check if all required vector store files exist"""
    required_files = ["index.faiss", "index.pkl"]
    return os.path.exists(VECTOR_STORE_DIR) and all(
        os.path.exists(os.path.join(VECTOR_STORE_DIR, f)) for f in required_files
    )


# Don't auto-create on import - let user run setup script first
if not vector_store_exists():
    logger.warning(
        f"Vector store not found at {VECTOR_STORE_DIR}. "
        "Please run 'python setup_vector_store.py' to create it."
    )