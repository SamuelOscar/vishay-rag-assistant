import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------------------------------------------------
# Custom equity research prompt
# ---------------------------------------------------
EQUITY_RESEARCH_PROMPT_TEMPLATE = """You are an expert equity research analyst specializing in Vishay Intertechnology (VSH).
You have deep knowledge of the company's financials, product segments, competitive positioning,
risk factors, and strategic outlook.

Use ONLY the information provided in the context below to answer the question.
If the context does not contain enough information to answer confidently, say:
"I don't have enough information in the available documents to answer that precisely."

Do NOT make up numbers, forecasts, or facts not grounded in the context.

When answering:
- Be concise but thorough
- Cite document type and year when referencing specific data (e.g. "According to the 2024 10-K...")
- Use financial terminology appropriate for an equity research context
- If the question involves numbers (revenue, margins, EPS), be precise

{chat_history}
---
Context from Vishay documents:
{context}
---

Question: {question}

Answer:"""


# ---------------------------------------------------
# Format retrieved docs into a single context string
# ---------------------------------------------------
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------
# Build the QA chain (returns chain + retriever)
# ---------------------------------------------------
def build_qa_chain(retriever):
    """
    Build a simple LCEL-based RAG chain using Groq.

    Args:
        retriever: A LangChain retriever (from retriever.py)

    Returns:
        A tuple of (chain, retriever) to be used with ask()
    """
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.0,  # deterministic — important for financial Q&A
    )

    prompt = ChatPromptTemplate.from_template(EQUITY_RESEARCH_PROMPT_TEMPLATE)

    chain = prompt | llm | StrOutputParser()

    return chain, retriever


# ---------------------------------------------------
# Ask a question with optional conversation history
# ---------------------------------------------------
def ask(chain_and_retriever, question: str, chat_history: list = None) -> dict:
    """
    Submit a question and return a grounded answer with sources.

    Args:
        chain_and_retriever: Tuple returned by build_qa_chain()
        question: The user's question
        chat_history: List of (question, answer) tuples from prior turns

    Returns:
        A dict with 'answer' and 'sources' keys
    """
    chain, retriever = chain_and_retriever

    # Retrieve relevant document chunks
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Format the last 3 conversation turns for context
    history_text = ""
    if chat_history:
        for past_q, past_a in chat_history[-3:]:
            history_text += f"Previous Q: {past_q}\nPrevious A: {past_a}\n\n"
        history_text = "Previous conversation:\n" + history_text

    # Run the chain
    answer = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": history_text,
    })

    # Deduplicate source filenames
    sources = list({
        doc.metadata.get("source", "Unknown source")
        for doc in docs
    })

    return {"answer": answer, "sources": sources}
