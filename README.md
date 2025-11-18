This project provides a comprehensive **Retrieval-Augmented Generation (RAG)** system built with **Streamlit**, **Ollama** (for local embeddings and LLM), and **HNSWLib** (for fast vector search).

It is split into two modular applications:

1.  **Ingestion System:** Processes PDF files, chunks the text, creates vector embeddings using Ollama, and builds a performant **HNSW index** for efficient storage and retrieval.
2.  **Chat System:** Loads the vector database, uses user queries to retrieve relevant document chunks, and feeds these chunks as context to an Ollama Large Language Model (LLM) to generate accurate, context-aware answers.

-----

## 🚀 Key Features

  * **Local-First RAG:** Uses **Ollama** for both local embedding (e.g., `nomic-embed-text`) and local LLM inference (e.g., `gemma3:latest`), ensuring data privacy and reducing API costs.
  * **Vector Search with HNSWLib:** Utilizes **Hierarchical Navigable Small World (HNSW)** indexing for highly efficient Approximate Nearest Neighbor (ANN) search, critical for fast retrieval.
  * **Modular Design:** Separate apps for **Ingestion** (data preparation) and **Chat** (inference).
  * **Advanced Retrieval:** Implements **Query Expansion** and **Similarity Threshold** filtering for improved retrieval accuracy.
  * **Streamlit UI:** User-friendly web interface for file upload, configuration, and conversational chat.

-----

## 🛠️ Prerequisites

Before running the application, you must have the following installed and set up:

1.  **Ollama:** Download and install the Ollama application.
2.  **Models:** Pull the required embedding and LLM models using the Ollama CLI.

<!-- end list -->

```bash
# Pull the embedding model (nomic-embed-text is the default)
ollama pull nomic-embed-text

# Pull the LLM model (gemma3:latest is the default)
ollama pull gemma3:latest
```

3.  **Python Packages:** Install the required Python libraries.

<!-- end list -->

```bash
pip install streamlit numpy ollama hnswlib pymupdf
```

-----

## 📁 Project Structure

The project consists of two main Python scripts and a data directory.

```
/rag-ollama-hnsw
├── app_ingestion.py   # PDF Processing, Chunking, Embedding, Indexing
├── app_chat.py        # Chat Interface, Retrieval, RAG Generation
├── requirements.txt   # (Optional) List of dependencies
└── /data/             # Stores the processed vector databases (HNSW index and data pickle)
    ├── merged_documents_data.pkl
    └── merged_documents_hnsw.bin
```

-----

## 1\. 📄 Ingestion System (`app_ingestion.py`)

This application is responsible for creating the searchable knowledge base from your documents.

### How to Run

```bash
streamlit run app_ingestion.py
```

### Usage Steps

1.  **Upload PDFs:** Upload one or more PDF files.
2.  **Select Processing Mode:**
      * **Merge into one knowledge base:** Combines all PDF text into a single, large document for comprehensive search. (Requires a custom name, e.g., `merged_documents`).
      * **Process separately:** Creates an individual knowledge base for each PDF.
3.  **Configure Settings (Optional):** Adjust `Chunk size`, `Overlap`, and HNSW optimization parameters (`ef_construction`, `M`).
4.  **Process & Embed:** Click the "🚀 Process & Embed Documents" button. The system will extract text, chunk it, create embeddings, and save the HNSW index and data files into the `/data` directory.

### Core Functions

  * `extract_text_from_pdf()`: Uses `PyMuPDF (fitz)` for robust text extraction.
  * `chunk_text()`: Splits text into overlapping chunks for better contextual retrieval.
  * `create_embeddings()`: Gets vector representations using the specified Ollama embedding model.
  * `create_hnsw_index()`: Builds the HNSW graph structure for fast similarity search.

-----

## 2\. 💬 Chat System (`app_chat.py`)

This application serves as the conversational interface for querying your ingested documents.

### How to Run

```bash
streamlit run app_chat.py
```

### Usage Steps

1.  **Load Document:** The app automatically loads the first available document from the `/data` directory upon startup. Use the sidebar to switch between or select your processed document.
2.  **Adjust Settings (Optional):** Use the sidebar settings to control the retrieval process:
      * **Chunks to retrieve (Top K):** How many chunks are passed to the LLM context.
      * **Similarity Threshold:** Filters out chunks below a certain relevance score.
3.  **Start Chatting:** Ask a question about the document in the chat input.

### Core Functions

  * `get_embedding()`: Gets the embedding of the user's query.
  * `expand_query()`: **New feature** that generates variations of the user's question to improve the chance of a good search match.
  * `search_similar_chunks()`: Performs the HNSW search, combines results from expanded queries, and filters by the similarity threshold.
  * `stream_answer()`: Constructs a RAG prompt using the retrieved context and streams the LLM's response from Ollama.

### RAG Workflow Diagram
