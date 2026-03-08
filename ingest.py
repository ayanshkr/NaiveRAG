import chromadb, itertools as it, logging, pymupdf, os, requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(filename="ingest.log", filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class DocumentManager:
    def __init__(self, url_path: str = os.getenv("URL_PATH"), storage_dir: str = os.getenv("STORAGE_DIR")):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        try:
            with open(url_path, 'r', encoding='utf-8') as f:
                self.urls = [line.strip().replace('\ufeff', '') for line in f if line.strip()]
            logger.info(f"Loaded {len(self.urls)} URLs from {url_path}")
        except Exception as e:
            logger.error(f"Failed to load URL file: {e}")
            self.urls = []

    def get_file_mappings(self):
        return [(url, os.path.join(self.storage_dir, url.split('/')[-1]), url.split('/')[-1]) for url in self.urls]

    def download(self):
        for url, path, name in self.get_file_mappings():
            if os.path.exists(path):
                logger.info(f"Skipping download: {name} already exists.")
                continue
            try:
                logger.info(f"Downloading: {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                with open(path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Successfully saved: {name}")
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")

class Processor:
    def __init__(self, chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000)), 
                 chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 100)),
                 seps: list[str] = ["\n\n","\n",".",","]):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=seps)

    def process_pdf(self, file_path: str, filename: str):
        logger.info(f"Processing PDF: {filename}")
        try:
            doc = pymupdf.open(file_path)
            chunk_count = 0
            for page_idx, page in enumerate(doc):
                offset = 0
                for chunk in self.splitter.split_text(page.get_text()):
                    chunk_count += 1
                    yield chunk, {"doc_name": filename, "page": page_idx + 1, "start": offset, "end": (offset := offset + len(chunk))}
            logger.info(f"Generated {chunk_count} chunks from {filename}")
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

class VectorStore:
    def __init__(self, db_dir: str = os.getenv("DB_DIR"), collection_name: str = os.getenv("COLLECTION_NAME")):
        self.client = chromadb.PersistentClient(path=db_dir)
        try:
            logger.info(f"Initializing collection: {collection_name}")
            self.collection = self.client.get_collection(collection_name)
            self.client.delete_collection(collection_name)
            self.collection = self.client.create_collection(collection_name)
            logger.info("Fresh collection created.")
        except Exception as e:
            logger.error(f"Error: {e}")
            self.collection = self.client.create_collection(collection_name)
            logger.info(f"Collection {collection_name} created.")

    def upload_batches(self, data_generator, batch_size: int = 5461):
        id_gen = (str(i) for i in it.count())
        for batch_idx, batch in enumerate(it.batched(data_generator, batch_size)):
            chunks, metas = zip(*batch)
            self.collection.add(documents=list(chunks),metadatas=list(metas),ids=list(it.islice(id_gen, len(chunks))))
            logger.info(f"Uploaded batch {batch_idx} ({len(chunks)} chunks).")

def main():
    logger.info("--- Starting Ingestion Process ---")
    doc_mgr, processor, store = DocumentManager(), Processor(), VectorStore()
    doc_mgr.download()
    def stream_all_data():
        for _, path, name in doc_mgr.get_file_mappings():
            if os.path.exists(path):
                yield from processor.process_pdf(path, name)
    store.upload_batches(stream_all_data())
    logger.info("--- Ingestion Complete ---")

if __name__ == "__main__":
    main()