import chromadb, ollama, os, re, logging
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(filename="ask.log", filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class RAGSearcher:
    def __init__(self, db_path: str = os.getenv("DB_DIR"), collection_name: str = os.getenv("COLLECTION_NAME"), model: str = os.getenv("MODEL")):
        self.model = model
        self.system_prompt = "Answer the question based only on the following context. Cite each piece of information using the format [chunk X]. If there is not enough information, answer 'Not found in the provided PDFs.'"
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Connected to ChromaDB at '{db_path}', collection: '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to connect to Vector Store: {e}")
            raise

    def retrieve(self, question: str, n_results: int = 6) -> Dict[str, Any]:
        logger.info(f"Searching for: '{question}'")
        results = self.collection.query(query_texts=[question], n_results=n_results)
        retrieved_ids = results.get('ids', [[]])[0]
        logger.info(f"Retrieved {len(retrieved_ids)} relevant chunks: {retrieved_ids}")
        return results

    def generate_answer(self, question: str, context: str) -> str:
        messages = [{'role': 'system', 'content': self.system_prompt},{'role': 'user', 'content': f"CONTEXT:\n{context}\n\nQUESTION: {question}"}]
        logger.info(f"Sending request to Ollama (model: {self.model})...")
        try:
            response = ollama.chat(model=self.model, messages=messages)
            logger.info("Received response from LLM.")
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return "Error: Could not generate an answer."

class CitationProcessor:
    def __init__(self, search_results: Dict[str, Any]):
        self.results = search_results
        self.ids = search_results['ids'][0]
        self.metas = search_results['metadatas'][0]
        self.docs = search_results['documents'][0]

    def format_context(self) -> str:
        return '\n\n'.join([f"Context from [Chunk {id}]:\n{doc}" for id, doc in zip(self.ids, self.docs)])

    def finalize_response(self, raw_text: str):
        new_response = raw_text
        used_citations = []
        matches = list(re.finditer(r'\[?[Cc]hunk\s+(\d+)[^\]]*\]?', raw_text))
        for match in sorted(matches, key=lambda x: x.start(), reverse=True):
            chunk_id = match.group(1)
            try:
                idx = self.ids.index(chunk_id)
                meta = self.metas[idx]
                citation_str = f"[{meta['doc_name']}, p.{meta['page']}, chunk:{chunk_id}]"
                new_response = new_response[:match.start()] + citation_str + new_response[match.end():]
                if citation_str not in used_citations:
                    used_citations.append(citation_str)
            except (ValueError, KeyError):
                logger.warning(f"LLM cited Chunk ID '{chunk_id}' but it wasn't in the retrieved context.")
                new_response = new_response[:match.start()] + "[Invalid Citation]" + new_response[match.end():]
        logger.info(f"Finalized response with {len(used_citations)} unique citations.")
        return new_response, used_citations

def main():
    try:
        searcher = RAGSearcher()
    except Exception as e:
        logger.error(f"An unexpected error occurred when initializing RAG Searcher: {e}")
        return
    n = 1
    while True:
        try:
            question = input(f"\n### Question {n}: ").strip()
            if not question:
                logger.info("Session ended by user.")
                break
            results = searcher.retrieve(question)
            processor = CitationProcessor(results)
            context = processor.format_context()
            raw_answer = searcher.generate_answer(question, context)
            final_text, citations_list = processor.finalize_response(raw_answer)
            print("\n" + "="*50)
            print(f"**Answer:**\n{final_text.strip()}")
            if citations_list:
                print("\n**Citations:**")
                for cit in citations_list:
                    print(f"• {cit}")
            print("\n**Retrieved Evidence:**")
            for i, (chunk_id, meta, doc) in enumerate(zip(processor.ids, processor.metas, processor.docs), 1):
                cit = f"[{meta['doc_name']}, p.{meta['page']}, chunk:{chunk_id}]"
                status = " [USED]" if cit in citations_list else " [UNUSED]"
                print(f"{i}. {cit}{status}")
                print(f"   \"{doc[:120].replace('\n', ' ')}...\"")
            print("="*50)
            n += 1
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.exception(f"An unexpected error occurred during Question {n}: {e}")

if __name__ == "__main__":
    main()