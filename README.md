LexBharat — Document-Grounded Legal Research Assistant

LexBharat is a Retrieval-Augmented Generation (RAG) system designed to answer legal questions strictly based on uploaded documents.
Instead of relying on a model’s general knowledge, LexBharat retrieves relevant sections from legal texts and generates answers grounded in those sources.

The goal is to explore how LLMs, embeddings, and vector databases can be combined to build reliable systems that minimize hallucinations and provide traceable, document-based responses.

Why LexBharat?

Large Language Models are powerful, but they often hallucinate when asked about specific documents.

LexBharat addresses this by implementing a document-grounded question answering pipeline:

Convert legal documents into semantic chunks

Store embeddings in a vector database

Retrieve the most relevant sections for a user query

Generate answers using only the retrieved context

This ensures that responses are anchored in the document itself.

System Architecture
PDF Document
      │
      ▼
Document Loader (PyPDF)
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation (HuggingFace)
      │
      ▼
Vector Storage (FAISS)
      │
      ▼
Query Retrieval
      │
      ▼
LLM Answer Generation (Ollama / LLaMA)

The LLM is not allowed to answer without context, reducing hallucination risk.

Key Features

Document Grounded Answers

Responses are generated using retrieved chunks from the uploaded document.

Semantic Search

FAISS enables similarity search across embedded document chunks.

Modular Architecture

Each component is isolated for experimentation:

Document Loading

Chunking

Embeddings

Vector Storage

Retrieval

LLM Generation

Local LLM Support

The system uses Ollama to run models locally, avoiding external API dependencies.

Tech Stack

Python

LangChain

HuggingFace Embeddings

FAISS Vector Database

Ollama (LLaMA models)

PyPDF for document parsing

Project Structure
backend/
│
├── app/
│   ├── core/
│   │   ├── loaders.py        # PDF ingestion and chunking
│   │   ├── embeddings.py     # HuggingFace embedding model
│   │   ├── vectorstore.py    # FAISS vector database
│   │   ├── retriever.py      # Semantic search
│   │   └── llm.py            # LLM answer generation
│   │
│   └── main.py               # Pipeline entry point
│
├── data/
│   └── raw/                  # Source documents
│
└── requirements.txt
Example Workflow

Add a legal document to data/raw/

Run the pipeline

Ask questions related to the document

Example:

Enter your question:
What was the main issue in the case?

Output:

Answer:
The main issue discussed in the document concerns the interpretation of personal liberty under Article 21 of the Constitution.

Source:
Relevant passages retrieved from the document.
Running the Project
1. Clone the repository
git clone https://github.com/siddhant1729/LeXBharat.git
cd LeXBharat
2. Create virtual environment
python -m venv venv

Activate:

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Run the pipeline

From the backend directory:

python -m app.main
Learning Goals

LexBharat was built as an exploration into:

How vector databases enable semantic search

How RAG reduces hallucinations

How LLMs can be constrained using retrieved context

Designing modular AI systems instead of simple wrappers

Future Directions

Potential extensions include:

Citation-aware answers with page references

Multi-document retrieval across legal cases

Confidence scoring for out-of-scope queries

LangGraph-based reasoning pipelines

Author

Siddhant Shaurya
Computer Science Undergraduate — JIIT Noida

Interested in:

AI Systems

Machine Learning

Backend Engineering

Intelligent developer tools

GitHub:
https://github.com/siddhant1729
