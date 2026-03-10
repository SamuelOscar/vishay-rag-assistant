"""
Vishay Intertechnology RAG Assistant
======================================
An equity research Q&A assistant powered by LangChain + FAISS + Groq (Llama 3).

Usage:
    python src/main.py

Make sure you have:
    1. A .env file with GROQ_API_KEY set
    2. Run `python src/ingest.py` first to build the vector store
"""

import os
import sys

from dotenv import load_dotenv
from retriever import get_retriever
from qa_chain import build_qa_chain, ask

load_dotenv()

# ---------------------------------------------------
# Display helpers
# ---------------------------------------------------
DIVIDER = "─" * 60
BANNER = """
╔══════════════════════════════════════════════════════════╗
║       Vishay Intertechnology  |  RAG Research Assistant  ║
║       Powered by LangChain + FAISS + Groq (Llama 3)      ║
╚══════════════════════════════════════════════════════════╝
"""

EXAMPLE_QUERIES = [
    "What was Vishay's total revenue in fiscal year 2024?",
    "What are the main product segments and their revenue contributions?",
    "What are the key risk factors mentioned in the most recent 10-K?",
    "How has Vishay's gross margin trended over the past three years?",
    "What is Vishay's strategy for growth in the automotive segment?",
    "What did management say about inventory levels in Q4 2025?",
    "Who are Vishay's main competitors?",
]


def print_welcome():
    print(BANNER)
    print("  Ask any question about Vishay Intertechnology based on filings,")
    print("  earnings reports, investor presentations, and press releases.\n")
    print("  Commands:")
    print("    'examples'  — show sample questions")
    print("    'clear'     — clear conversation history")
    print("    'quit'      — exit the assistant\n")
    print(DIVIDER)


def print_examples():
    print("\n📋 Example questions you can ask:\n")
    for i, q in enumerate(EXAMPLE_QUERIES, 1):
        print(f"  {i}. {q}")
    print()


def format_answer(answer: str, sources: list) -> str:
    """Format and display the answer with source citations."""
    # Normalize line endings (fixes Windows \r overwrite issue)
    answer = answer.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Indent each line for clean display
    indented = "\n   ".join(answer.split("\n"))
    output = f"\n💬 Answer:\n   {indented}\n"

    if sources:
        output += "\n📄 Sources used:\n"
        for src in sorted(sources):
            output += f"   • {src}\n"

    return output


# ---------------------------------------------------
# Main conversation loop
# ---------------------------------------------------
def main():
    print_welcome()

    # 1. Load the retriever (requires ingest.py to have been run)
    print("⏳ Loading vector store...")
    try:
        retriever = get_retriever(k=7)
        print("✅ Vector store loaded successfully.\n")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Run this first:  python src/ingest.py\n")
        sys.exit(1)

    # 2. Build the QA chain
    chain_and_retriever = build_qa_chain(retriever)

    # 3. Session memory: list of (question, answer) tuples
    chat_history = []

    print(f"{DIVIDER}\n")

    # 4. Interactive Q&A loop
    while True:
        try:
            user_input = input("🔍 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!\n")
            break

        if user_input.lower() == "examples":
            print_examples()
            continue

        if user_input.lower() == "clear":
            chat_history = []
            print("\n🗑️  Conversation history cleared.\n")
            continue

        # 5. Query the chain
        print(f"\n{DIVIDER}")
        print("⏳ Retrieving and generating answer...")

        try:
            result = ask(chain_and_retriever, user_input, chat_history)

            # Save this exchange to memory
            chat_history.append((user_input, result["answer"]))

            print(format_answer(result["answer"], result["sources"]))
        except Exception as e:
            print(f"\n❌ Error generating answer: {e}\n")

        print(DIVIDER + "\n")


if __name__ == "__main__":
    main()
