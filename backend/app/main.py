from app.core.loaders import load_and_split_pdf
from app.core.embeddings import get_embeddings
from app.core.vectorstore import create_faiss_store
from app.core.retriever import retrieve_from_Faiss
from app.core.llm import generate_answer


def main():
    # 1. Load and split the document
    chunks = load_and_split_pdf("../data/raw/sample.pdf")

    # 2. Initialize embeddings
    embeddings = get_embeddings()

    # 3. Create FAISS vector store
    vectorstore = create_faiss_store(chunks, embeddings)

    # 4. User query
    query = input("Enter your question: ").strip()

    if not query:
        print("No query provided.")
        return

    # 5. Retrieve relevant chunks
    retrieved_chunks = retrieve_from_Faiss(vectorstore, query, k=4)

    if not retrieved_chunks:
        print("No relevant context found in the document.")
        return

    # 6. Generate grounded answer
    answer = generate_answer(query, retrieved_chunks)

    # 7. Output
    print("\n===== ANSWER =====")
    print(answer)


if __name__ == "__main__":
    main()
