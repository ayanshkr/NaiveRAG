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
URL_PATH           | "urls.txt"          | Source PDF list  | Targets high-density technical NASA/NIST standards.                  |
STORAGE_DIR        | "docs"              | PDF cache folder | Local repository for downloaded technical PDFs.                      |
CHUNK_SIZE         | 1200                | Text segment size| Captures complex requirements without losing local context.          |
CHUNK_OVERLAP      | 200                 | Context overlap  | Maintains continuity for concepts spanning across pages.              |
DB_DIR             | "chroma_db"         | Vector DB path   | Persistent local storage for document embeddings.                     |
COLLECTION_NAME    | "all_documents"     | DB index name    | Standard namespace for the vectorized handbook library.              |
MODEL              | "gemma3:4b-it-qat"  | Local LLM model  | Highly efficient for strict citation following in a local environment.|

------------------------------------------------------------
2. TECHNICAL DESIGN DECISIONS
------------------------------------------------------------
These engineering choices prioritize precision and 
verifiability in technical domains:

[1] PAGE-AWARE INGESTION:
    `ingest.py` utilizes PyMuPDF to extract text on a per-page 
    basis. This allows the system to bake the 'page' number 
    directly into the vector metadata for verifiable citations.

[2] CONTEXTUAL CHUNKING:
    A 1200-character chunk size was selected to capture 
    complete technical units (like safety requirements) 
    while the 200-character overlap ensures concepts are not 
    severed at arbitrary character limits.

[3] REGEX-ANCHORED CITATIONS:
    `ask.py` does not rely on LLM memory for sourcing. It 
    forces the model to use [chunk X] placeholders, which 
    the Python backend then maps to real metadata via 
    regular expressions to prevent source hallucination.

[4] BATCHED VECTORIZATION:
    To handle massive PDF libraries, the system uses 
    itertools.batched (size 5461) to commit data to 
    ChromaDB, preventing memory overflows during ingestion.

------------------------------------------------------------
3. INSTALLATION & SETUP
------------------------------------------------------------
Ensure you have Python 3.10+ and Ollama installed.

[1] Install dependencies:
    $ pip install -r requirements.txt

[2] Pull the local LLM:
    $ ollama pull gemma3:4b-it-qat

------------------------------------------------------------
4. USAGE FLOW
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
5. LOGGING & DEBUGGING
------------------------------------------------------------
All activity is recorded in 'ingest.log' and 'ask.log' respectively.
- Check ingest.log if a download fails or text extraction stalls.
- Check ask.log to verify the precision of LLM source mapping.

------------------------------------------------------------
6. DATA FLOW ARCHITECTURE
------------------------------------------------------------

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

    subgraph Ingest [ingest.py Engine]
        direction TB
        D_MGR[DocumentManager]:::logic
        PROC[Processor]:::logic
        V_STORE[VectorStore]:::logic
    end

    subgraph Storage [Storage]
        PDFS[docs/ folder]:::destination
        CHROMA[(ChromaDB)]:::destination
    end

    subgraph Query [ask.py Engine]
        direction TB
        SRCH[RAGSearcher]:::logic
        C_PROC[CitationProcessor]:::logic
        OLLAMA[Ollama API]:::logic
    end

    %% --- Data Flow Connections ---
    U_TXT -- "Raw URL List" --> D_MGR
    ENV -- "CHUNK_SIZE / MODEL" --> D_MGR & PROC & SRCH & OLLAMA
    
    D_MGR -- "Binary Stream" --> PDFS
    D_MGR -- "Local File Path" --> PROC
    
    PROC -- "Page Text + Metadata" --> SPLIT[Text Splitter]:::logic
    SPLIT -- "1200 Char Chunks" --> V_STORE
    V_STORE -- "Batch ID + Vector + Meta" --> CHROMA
    
    D_MGR -. "Success/Error Logs" .-> I_LOG[ingest.log]:::logs

    USER -- "Plaintext Question" --> SRCH
    CHROMA -- "Top 6 Chunks + Metadata" --> SRCH
    
    SRCH -- "Context Block List" --> C_PROC
    C_PROC -- "Structured System Prompt" --> OLLAMA
    
    OLLAMA -- "Raw String w/ Chunk ID" --> C_PROC
    C_PROC -- "Page-Verified Answer" --> TERM[Terminal Display]:::destination
    
    C_PROC -. "Mapping Stats" .-> A_LOG[ask.log]:::logs
