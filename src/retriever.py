import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------
load_dotenv()

VECTORSTORE_PATH = "vectorstore/faiss_index"


# ---------------------------------------------------
# Load the FAISS vector store from disk
# ---------------------------------------------------
def load_vectorstore():
    """Load the pre-built FAISS index from disk."""
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTORSTORE_PATH}'.\n"
            "Please run: python src/ingest.py"
        )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


# ---------------------------------------------------
# Build a retriever from the vector store
# ---------------------------------------------------
def build_retriever(vectorstore, k: int = 5):
    """
    Create a retriever from the FAISS vectorstore.

    Args:
        vectorstore: A loaded FAISS vectorstore instance.
        k: Number of top document chunks to retrieve per query.

    Returns:
        A LangChain retriever object.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever


# ---------------------------------------------------
# Combined loader: get retriever in one call
# ---------------------------------------------------
def get_retriever(k: int = 5):
    """
    Convenience function: load the vectorstore and return a retriever.

    Args:
        k: Number of chunks to retrieve per query.

    Returns:
        A configured LangChain retriever.
    """
    vectorstore = load_vectorstore()
    return build_retriever(vectorstore, k=k)
