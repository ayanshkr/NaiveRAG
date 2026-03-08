import chromadb, codecs, ollama, os, re
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

class RAGSearcher:
  def __init__(self, db_path: str = os.getenv("DB_DIR","chroma_db"), collection_name: str = os.getenv("COLLECTION_NAME","all_documents"), model: str = os.getenv("MODEL","gemma3:4b-it-qat")):
    self.client = chromadb.PersistentClient(path=db_path)
    self.collection = self.client.get_collection(collection_name)
    self.model = model
    self.system_prompt = "Answer the question based only on the following context. Cite each piece of information using the format [chunk X]. If there is not enough information, answer 'Not found in the provided PDFs.'"
  def retrieve(self, question: str, n_results: int = 6) -> Dict[str, Any]:
    return self.collection.query(query_texts=[question], n_results=n_results)
  def generate_answer(self, question: str, context: str) -> str:
    messages = [{'role': 'system', 'content': self.system_prompt},{'role': 'user', 'content': f"{context}\n\n{question}"}]
    return ollama.chat(model=self.model, messages=messages)['message']['content']

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
    for match in sorted(re.finditer(r'\[?[Cc]hunk\s+(\d+)[^\]]*\]?', raw_text), key = lambda x: x.start(), reverse=True):
      try:
        meta = self.metas[self.ids.index((chunk_id := match.group(1)))]
        citation_str = f"[{meta['doc_name']}, p.{meta['page']}, chunk:{chunk_id}]"
        new_response = new_response[:match.start()] + citation_str + new_response[match.end():]
        if citation_str not in used_citations:
          used_citations.append(citation_str)
      except (ValueError, KeyError):
        new_response = new_response[:match.start()] + new_response[match.end():]
    return new_response, used_citations

def main():
  searcher = RAGSearcher()
  while True:
    if not (question := input("\nEnter question: ").strip()): break
    context = (processor := CitationProcessor(searcher.retrieve(question))).format_context()
    raw_answer = searcher.generate_answer(question, context)
    final_text, citations_list = processor.finalize_response(raw_answer)
    print("\n",codecs.decode(final_text, 'unicode-escape').strip(),"\n\nCitations:")
    for cit in citations_list:
      print(f"- {cit}")
    print("\nRetrieved Evidence:")
    for i, (chunk_id, meta, doc) in enumerate(zip(processor.ids, processor.metas, processor.docs)):
      if (cit := f"[{meta['doc_name']}, p.{meta['page']}, chunk:{chunk_id}]") in citations_list:
        print(f"{i+1}. {cit}")
        print(f"   \"{doc[:200].replace('\n', ' ').strip()}...\"\n   (showing first 200 characters)")

if __name__ == "__main__":
  main()