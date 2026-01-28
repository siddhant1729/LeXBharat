from langchain_ollama import OllamaLLM


llm = OllamaLLM(model="llama3")


def generate_answer(query: str, chunks):
    """
    Generates an answer using only the provided chunks.
    """

    context = "\n\n".join(
        chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        for chunk in chunks
    )

    prompt = f"""
You are a legal assistant.

Answer the question using ONLY the context below.
If the answer is not present, say:
"The document does not contain this information."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response
