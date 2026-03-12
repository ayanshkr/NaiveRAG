🚀 UNIVERSAL PDF-RAG PIPELINE
============================================================

A high-performance, local-first Retrieval-Augmented 
Generation (RAG) system for technical document analysis.

------------------------------------------------------------
1. PROJECT CONFIGURATION (.env)
------------------------------------------------------------
The system is controlled via the following environment 
variables. Ensure these are set in your .env file:

PROPERTY           | VALUE               | PURPOSE          | REASON FOR DEFAULT                                                   |
-------------------|---------------------|------------------|----------------------------------------------------------------------|
URL_PATH           | "urls.txt"          | Source PDF list  | Short name and long pdfs with serious subjects for testing           |
STORAGE_DIR        | "docs"              | PDF cache folder | Short and descriptive name                                           |
CHUNK_SIZE         | 1200                | Text segment size| Enough to capture units of information while not overwhelming a user |
CHUNK_OVERLAP      | 200                 | Context overlap  | Enough to break distinct ideas while not breaking contextual units   |
DB_DIR             | "chroma_db"         | Vector DB path   | Short and descriptive name                                           |
COLLECTION_NAME    | "all_documents"     | DB index name    | Short and descriptive name                                           |
MODEL              | "gemma3:4b-it-qat"  | Local LLM model  | Context window can fit one prompt and runs on most computers decently|

------------------------------------------------------------
2. INSTALLATION & SETUP
------------------------------------------------------------
Ensure you have Python 3.10+ and Ollama installed.

[1] Install dependencies:
    $ pip install -r requirements.txt

[2] Pull the local LLM:
    $ ollama pull gemma3:4b-it-qat

------------------------------------------------------------
3. USAGE FLOW
------------------------------------------------------------

PHASE A: INGESTION
Run the command: $ python ingest.py
- Downloads PDFs from URLs provided in urls.txt.
- Extracts text page-by-page using PyMuPDF.
- Stores 1200-character vectors into ChromaDB.

PHASE B: QUERYING
Run the command: $ python ask.py
- Enter your question when prompted.
- The system retrieves the 6 most relevant document chunks.
- The AI generates an answer with page-level citations.

------------------------------------------------------------
4. DATA SOURCES (urls.txt)
------------------------------------------------------------
This system is pre-configured to handle technical standards 
from NASA and NIST, including:
- NASA Systems Engineering Handbook
- NIST SP 800-53 Security Controls
- NASA Schedule Management Standards

------------------------------------------------------------
5. LOGGING & DEBUGGING
------------------------------------------------------------
All activity is recorded in 'ingest.log' and 'ask.log' respectively.
- Check this file if a download fails.
- Check this file to verify LLM citation accuracy.

### NaiveRAG Data Flow Architecture

```mermaid
graph LR
    %% --- Style Definitions ---
    classDef source fill:#f472b6,stroke:#be185d,stroke-width:2px,color:#fff;
    classDef logic fill:#38bdf8,stroke:#0369a1,stroke-width:2px,color:#fff;
    classDef destination fill:#4ade80,stroke:#15803d,stroke-width:2px,color:#fff;
    classDef logs fill:#64748b,stroke:#334155,stroke-dasharray: 5 5,color:#fff;

    %% --- Components ---
    subgraph Sources [Data Sources]
        U_TXT[urls.txt]:::source
        ENV[.env]:::source
        USER[User Prompt]:::source
    end

    subgraph Ingest [ingest.py]
        direction TB
        D_MGR[DocumentManager]:::logic
        PROC[Processor]:::logic
        V_STORE[VectorStore]:::logic
    end

    subgraph Storage [Storage]
        PDFS[docs/ folder]:::destination
        CHROMA[(ChromaDB)]:::destination
    end

    subgraph Query [ask.py]
        direction TB
        SRCH[RAGSearcher]:::logic
        C_PROC[CitationProcessor]:::logic
        OLLAMA[Ollama API]:::logic
    end

    %% --- Data Flow Connections ---

    %% Ingestion Path
    U_TXT -- "Raw URL List" --> D_MGR
    ENV -- "CHUNK_SIZE / MODEL" --> D_MGR & PROC & SRCH & OLLAMA
    
    D_MGR -- "Binary Stream" --> PDFS
    D_MGR -- "Local File Path" --> PROC
    
    PROC -- "Page Text + Metadata" --> SPLIT[Text Splitter]:::logic
    SPLIT -- "1200 Char Chunks" --> V_STORE
    V_STORE -- "Batch ID + Vector + Meta" --> CHROMA
    
    D_MGR -. "Success/Error Logs" .-> I_LOG[ingest.log]:::logs

    %% Query Path
    USER -- "Plaintext Question" --> SRCH
    CHROMA -- "Top 6 Chunks + Metadata" --> SRCH
    
    SRCH -- "Context Block List" --> C_PROC
    C_PROC -- "Structured System Prompt" --> OLLAMA
    
    OLLAMA -- "Raw String w/ Chunk ID" --> C_PROC
    C_PROC -- "Page-Verified Answer" --> TERM[Terminal Display]:::destination
    
    C_PROC -. "Mapping Stats" .-> A_LOG[ask.log]:::logs
