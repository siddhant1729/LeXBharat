

def retrieve_from_Faiss(vectorstore,query :str,k=4):
   
    results = vectorstore.similarity_search(query, k=k)

    return results