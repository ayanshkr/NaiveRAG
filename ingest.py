import chromadb, itertools as it, pymupdf, os, requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

class DocumentManager:
    def __init__(self, url_path: str = os.getenv("URL_PATH", "urls.txt"), storage_dir: str = os.getenv("STORAGE_DIR", "docs")):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.urls = [line.strip().replace('\ufeff', '') for line in open(url_path, 'r', encoding='utf-8') if line.strip()]
    def get_file_mappings(self):
        return [(url, os.path.join(self.storage_dir, url.split('/')[-1]), url.split('/')[-1]) for url in self.urls]
    def download(self):
        for url, path, _ in it.filterfalse(lambda out: os.path.exists(out[1]),self.get_file_mappings()):
            open(path, 'wb').write(requests.get(url).content)

class Processor:
    def __init__(self, chunk_size: int = os.getenv("CHUNK_SIZE",1200), chunk_overlap: int = os.getenv("CHUNK_OVERLAP",200),seps: list[str] = ["\n\n","\n",".",","," "]):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=seps)
    def process_pdf(self, file_path: str, filename: str):
        try:
            for page_idx, page in enumerate(pymupdf.open(file_path)):
                offset = 0
                for chunk in self.splitter.split_text(page.get_text()):
                    yield chunk, {"doc_name": filename, "page": page_idx + 1, "start": offset, "end": (offset := offset + len(chunk))}
        except pymupdf.EmptyFileError:
            print("Well... fucking shit")

class VectorStore:
    def __init__(self, db_dir: str = os.getenv("DB_DIR","chroma_db"), collection_name: str = os.getenv("COLLECTION_NAME","all_documents")):
        self.client = chromadb.PersistentClient(path=db_dir)
        try:
            self.collection = self.client.create_collection(collection_name)
        except (chromadb.errors.InternalError, Exception):
            self.client.delete_collection(collection_name)
            self.collection = self.client.create_collection(collection_name)
    def upload_batches(self, data_generator, batch_size: int = 5461):
        id_gen = (str(i) for i in it.count())
        for batch in it.batched(data_generator, batch_size):
            chunks, metas = zip(*batch)
            self.collection.add(documents=list(chunks),metadatas=list(metas),ids=list(it.islice(id_gen, len(chunks))))

def main():
    doc_mgr, processor, store = DocumentManager(), Processor(), VectorStore()
    doc_mgr.download()
    def stream_all_data():
        for _, path, name in doc_mgr.get_file_mappings():
            yield from processor.process_pdf(path, name)
    store.upload_batches(stream_all_data())
    print("Ingestion complete.")

if __name__ == "__main__":
    main()