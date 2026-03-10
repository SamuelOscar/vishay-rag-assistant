import os
import re
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# 2. Paths
# ---------------------------------------------------
RAW_DATA_PATH = "data/raw"
VECTORSTORE_PATH = "vectorstore/faiss_index"


# ---------------------------------------------------
# 3. Basic text cleaning function
# ---------------------------------------------------
def clean_text(text):
    # Remove excessive newlines
    text = re.sub(r"\n{2,}", "\n", text)
    # Remove excessive spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------
# 4. Add metadata based on filename
# ---------------------------------------------------
def extract_metadata(filename):
    metadata = {}

    # Example filename: VSH_10K_2024.html
    if "10K" in filename:
        metadata["type"] = "10-K"
    elif "10Q" in filename:
        metadata["type"] = "10-Q"
    elif "transcript" in filename.lower():
        metadata["type"] = "Earnings Transcript"
    else:
        metadata["type"] = "Unknown"

    # Extract year from filename
    year_match = re.search(r"20\d{2}", filename)
    if year_match:
        metadata["year"] = year_match.group()

    metadata["source"] = filename
    metadata["company"] = "Vishay Intertechnology"

    return metadata


# ---------------------------------------------------
# 5. Load documents from folder
# ---------------------------------------------------
def load_documents():
    documents = []
    skipped = []

    for file in os.listdir(RAW_DATA_PATH):
        # Use absolute path to avoid issues with relative path resolution
        file_path = os.path.abspath(os.path.join(RAW_DATA_PATH, file))

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".html"):
            loader = BSHTMLLoader(file_path, open_encoding="utf-8")
        else:
            continue

        try:
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata.update(extract_metadata(file))
                documents.append(doc)
            print(f"  ✓ {file}")
        except Exception as e:
            print(f"  ✗ Skipped (error): {file[:60]}... → {type(e).__name__}")
            skipped.append(file)

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s) due to errors.")

    return documents


# ---------------------------------------------------
# 6. Split documents into chunks
# ---------------------------------------------------
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    return text_splitter.split_documents(documents)


# ---------------------------------------------------
# 7. Create and save FAISS index
# ---------------------------------------------------
def create_vectorstore(chunks):
    print("Creating embeddings (local model, no API cost)...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("Saving FAISS index...")
    vectorstore.save_local(VECTORSTORE_PATH)

    print("Ingestion complete!")


# ---------------------------------------------------
# 8. Main Execution
# ---------------------------------------------------
if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    print("Splitting into chunks...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    create_vectorstore(chunks)
