from langchain_community.vectorstores import FAISS

def create_faiss_store(chunks, embeddings):
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vectorstore
