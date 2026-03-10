<div align="center">

<br/>

```
██╗     ███████╗██╗  ██╗██████╗ ██╗  ██╗ █████╗ ██████╗  █████╗ ████████╗
██║     ██╔════╝╚██╗██╔╝██╔══██╗██║  ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝
██║     █████╗   ╚███╔╝ ██████╔╝███████║███████║██████╔╝███████║   ██║   
██║     ██╔══╝   ██╔██╗ ██╔══██╗██╔══██║██╔══██║██╔══██╗██╔══██║   ██║   
███████╗███████╗██╔╝ ██╗██████╔╝██║  ██║██║  ██║██║  ██║██║  ██║   ██║   
╚══════╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
```

### Document-Grounded Legal Research Assistant

*Ask questions. Get answers. Traced back to the source.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-00599C?style=flat-square)](https://faiss.ai)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=flat-square)](https://ollama.ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)

</div>

---

## 📌 What is LexBharat?

LexBharat is a **Retrieval-Augmented Generation (RAG)** system built to answer legal questions — strictly from documents you provide.

No hallucinations. No general knowledge drift. Every answer is **traceable back to a source passage**.

> Built to explore how LLMs, embeddings, and vector databases can be combined to create reliable, document-grounded AI systems.

---

## 🧠 The Problem

Large Language Models are powerful — but they hallucinate when asked about specific documents. They blend training data with your input, producing confident but ungrounded answers.

**LexBharat addresses this directly:**

| Without RAG | With LexBharat |
|---|---|
| LLM answers from general training | LLM answers only from your document |
| Hard to trace where the answer came from | Every answer is grounded in retrieved passages |
| Hallucinations are common | Out-of-scope queries are rejected |

---

## ⚙️ System Architecture

```
📄 PDF Document
      │
      ▼
 Document Loader (PyPDF)
      │
      ▼
 Text Chunking
  [500–1000 token segments]
      │
      ▼
 Embedding Generation
  [HuggingFace Sentence Transformers]
      │
      ▼
 Vector Storage (FAISS)
      │
      ▼
 Query → Semantic Retrieval
  [Top-3 relevant passages]
      │
      ▼
 LLM Answer Generation
  [Ollama / LLaMA — context-only mode]
      │
      ▼
 ✅ Grounded Answer + Source Reference
```

> **Key constraint:** The LLM is not permitted to answer without retrieved context — hallucination risk is structurally reduced.

---

## ✨ Key Features

**📎 Document-Grounded Answers**  
Responses are generated exclusively from retrieved chunks of your uploaded document.

**🔍 Semantic Search via FAISS**  
Similarity search across embedded document chunks finds relevant passages even when phrasing differs.

**🧩 Modular Architecture**  
Each component is cleanly isolated — swap out any layer independently:

```
Document Loading → Chunking → Embeddings → Vector Store → Retrieval → LLM
```

**🖥️ Local LLM Support**  
Runs entirely offline via Ollama. No external API keys. No data leaves your machine.

**🚫 Strict Out-of-Scope Handling**  
When no relevant context is found, LexBharat refuses to answer rather than guessing.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| RAG Framework | LangChain |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Database | FAISS |
| LLM Runtime | Ollama (LLaMA models) |
| PDF Parsing | PyPDF |

---

## 📁 Project Structure

```
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
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/siddhant1729/LeXBharat.git
cd LeXBharat
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline
```bash
# From the backend directory
python -m app.main
```

---

## 💬 Example Workflow

1. Drop a legal document into `data/raw/`
2. Run the pipeline
3. Ask a question about it

```
Enter your question: What was the main issue in the case?

Answer:
The main issue discussed in the document concerns the interpretation
of personal liberty under Article 21 of the Constitution.

Source:
Relevant passages retrieved from pages 3–4 of the document.
```

---

## 🎯 Learning Goals

LexBharat was built as a hands-on exploration of:

- How vector databases enable semantic search over large documents
- How RAG structurally reduces LLM hallucinations
- How to constrain LLM generation using retrieved context
- Designing modular, composable AI pipelines

---

## 🔭 Future Directions

- [ ] Citation-aware answers with page number references  
- [ ] Multi-document retrieval across related legal cases  
- [ ] Confidence scoring for out-of-scope query detection  
- [ ] LangGraph-based multi-step reasoning pipelines  
- [ ] Web interface for document upload and Q&A  

---

## 👤 Author

**Siddhant Shaurya**  
Computer Science Undergraduate — JIIT Noida

Interested in AI Systems · Machine Learning · Backend Engineering · Intelligent Developer Tools

[![GitHub](https://img.shields.io/badge/GitHub-siddhant1729-181717?style=flat-square&logo=github)](https://github.com/siddhant1729)

---

<div align="center">
<sub>Built with curiosity. Grounded in documents.</sub>
</div>
