

def retrieve_from_Faiss(vectorstore,query :str):
   
    results = vectorstore.similarity_search(query, k=3)

    return results