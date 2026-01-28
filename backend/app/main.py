from app.core.loaders import load_and_split_pdf
from app.core.embeddings import get_embeddings
from app.core.vectorstore import create_faiss_store
from app.core.retriever import retrieve_from_Faiss
from app.core.llm import generate_answer
def main():
    chunks = load_and_split_pdf("../data/raw/sample.pdf")
    embeddings = get_embeddings()
    vectorstore = create_faiss_store(chunks, embeddings)

    query = "What is the main issue discussed in the document?"
    results = retrieve_from_Faiss(vectorstore, query)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:500])

if __name__ == "__main__":
    main()
