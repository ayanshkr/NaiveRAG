🚀 UNIVERSAL PDF-RAG PIPELINE
============================================================

A high-performance, local-first Retrieval-Augmented 
Generation (RAG) system for technical document analysis.

------------------------------------------------------------
1. PROJECT CONFIGURATION (.env)
------------------------------------------------------------
The system is controlled via the following environment 
variables. Ensure these are set in your .env file:

PROPERTY           | VALUE               | PURPOSE
-------------------|---------------------|------------------
URL_PATH           | "urls.txt"          | Source PDF list
STORAGE_DIR        | "docs"              | PDF cache folder
CHUNK_SIZE         | 1200                | Text segment size
CHUNK_OVERLAP      | 200                 | Context overlap
DB_DIR             | "chroma_db"         | Vector DB path
COLLECTION_NAME    | "all_documents"     | DB index name
MODEL              | "gemma3:4b-it-qat"  | Local LLM model

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
