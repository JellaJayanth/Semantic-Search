
import argparse

from search_engine import DocumentSearchEngine

SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Remote Work Policy",
        "text": (
            "Employees may work from home up to three days per week with manager "
            "approval. All remote employees must be reachable during core hours, "
            "9am to 3pm local time, and attend the weekly team sync. Equipment "
            "stipends of $500 per year are available for home office setup, "
            "covering monitors, chairs, and ergonomic keyboards. Requests for "
            "fully remote arrangements are reviewed case by case by HR."
        ),
    },
    {
        "id": "doc2",
        "title": "Expense Reimbursement Guide",
        "text": (
            "Business travel expenses, including flights, hotels, and meals, are "
            "reimbursed within 30 days of submitting a report through the finance "
            "portal. Meal reimbursement is capped at $75 per day domestically and "
            "$100 internationally. Receipts are required for any expense over $25. "
            "Alcohol is not reimbursable except at approved client dinners."
        ),
    },
    {
        "id": "doc3",
        "title": "Onboarding Checklist for New Hires",
        "text": (
            "New employees complete IT setup on day one, including laptop "
            "provisioning and account creation. HR conducts a benefits orientation "
            "in the first week covering health insurance, 401k enrollment, and "
            "paid time off accrual. Managers should schedule a 30-60-90 day check-in "
            "to review goals and provide feedback during the ramp-up period."
        ),
    },
    {
        "id": "doc4",
        "title": "Data Retention and Privacy Policy",
        "text": (
            "Customer data is retained for 24 months after account closure unless "
            "otherwise required by law, after which it is permanently deleted. "
            "Personal data requests under privacy regulations such as GDPR must be "
            "fulfilled within 30 days. Access to production customer data is "
            "restricted to engineers on the on-call rotation and logged for audit."
        ),
    },
    {
        "id": "doc5",
        "title": "Vector Database Overview",
        "text": (
            "A vector database indexes high-dimensional embeddings and supports "
            "approximate nearest neighbor search, enabling semantic retrieval over "
            "unstructured text, images, or audio. Popular systems include FAISS, "
            "which is a library rather than a standalone server, and ChromaDB, "
            "Pinecone, and Weaviate, which run as managed or self-hosted databases "
            "with built-in persistence and metadata filtering."
        ),
    },
]

DEMO_QUERIES = [
    "How many days can I work from home?",
    "What's the daily limit for meal expenses on a business trip?",
    "What happens to customer data after they close their account?",
    "How does similarity search work for embeddings?",
    "What does a new employee do in their first week?",
]


def main():
    parser = argparse.ArgumentParser(description="Semantic document search demo")
    parser.add_argument("--vector", choices=["faiss", "chroma", "numpy"], default="faiss")
    parser.add_argument("--embedder", choices=["auto", "sentence-transformers", "tfidf"], default="auto")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    print(f"Building index  |  vector_backend={args.vector}  embedder_backend={args.embedder}")
    engine = DocumentSearchEngine(
        embedder_backend=args.embedder,
        vector_backend=args.vector,
        chunk_size=120,   # small on purpose: sample docs are short paragraphs
        chunk_overlap=20,
    )

    n_chunks = engine.ingest(SAMPLE_DOCUMENTS)
    print(f"Indexed {len(SAMPLE_DOCUMENTS)} documents -> {n_chunks} chunks "
          f"(embedder actually used: {type(engine.embedder).__name__})\n")

    for query in DEMO_QUERIES:
        print(f"QUERY: {query}")
        results = engine.search(query, top_k=args.top_k)
        for rank, r in enumerate(results, start=1):
            snippet = r["text"] if len(r["text"]) <= 140 else r["text"][:140] + "..."
            print(f"  {rank}. [{r['score']:.3f}] {r['doc_title']}  -  {snippet}")
        print()


if __name__ == "__main__":
    main()
