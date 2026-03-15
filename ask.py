import chromadb, logging, ollama, os, re, time
from chromadb.utils import embedding_functions
from typing import Dict, Any
from dotenv import load_dotenv

os.environ["TRANSFORMERS_OFFLINE"] = "1"
load_dotenv()
logging.basicConfig(filename="ask.log", filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class RAGSearcher:
    def __init__(self, db: str = os.getenv("DB_DIR",''), cl: str = os.getenv("COLLECTION_NAME",''), m: str = os.getenv("MODEL",'')):
        self.m, self.p = m, "Answer based only on context. Cite as [chunk X]. If missing, say 'Not found in the provided PDFs.'"
        logger.info(f"Init Searcher | Model: {m} | DB: {db}")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=os.getenv("EMBEDDING_MODEL",''), device="cuda")
        self.coll = chromadb.PersistentClient(path=db).get_collection(cl, embedding_function=ef) # type: ignore
        logger.info("Searcher ready.")

    def retrieve(self, q: str, n: int = 6) -> Dict[str, Any]:
        t = time.time()
        res = self.coll.query(query_texts=[q], n_results=n)
        logger.info(f"Retrieval: {len(res['ids'][0])} chunks in {time.time()-t:.4f}s")
        return dict(res)

    def generate(self, q: str, ctx: str) -> str:
        msgs = [{'role': 'system', 'content': self.p}, {'role': 'user', 'content': f"CONTEXT:\n{ctx}\n\nQUESTION: {q}"}]
        logger.info(f"Ollama Start: {self.m} | Context: {len(ctx)} chars")
        t = time.time()
        try:
            r = ollama.chat(model=self.m, messages=msgs, options={'keep_alive': '10m'})
            d, ec, ed = time.time()-t, r.get('eval_count', 0), r.get('eval_duration', 1)/1e9
            logger.info(f"Ollama End | Total: {d:.2f}s | Pre-fill: {r.get('prompt_eval_duration',0)/1e9:.2f}s | {ec} tks @ {ec/ed:.2f} tps")
            return r['message']['content']
        except Exception as e:
            logger.error(f"Ollama failed: {e}"); return "Error: Could not generate answer."

class CitationProcessor:
    def __init__(self, res: Dict[str, Any]): self.ids, self.mts, self.dcs = res['ids'][0], res['metadatas'][0], res['documents'][0]

    def format_ctx(self) -> str: return '\n\n'.join([f"Context from [chunk {i}]:\n{d}" for i, d in zip(self.ids, self.dcs)])

    def finalize(self, raw: str):
        res, used = raw, []
        for m in sorted(re.finditer(r'\[?[Cc]hunk\s+(\d+)[^\]]*\]?', raw), key=lambda x: x.start(), reverse=True):
            cid = m.group(1)
            try:
                mt = self.mts[self.ids.index(cid)]; ct = f"[{mt['doc_name']}, p.{mt['page']}, chunk:{cid}]"
                res = res[:m.start()] + ct + res[m.end():]
                if ct not in used: used.append(ct)
            except: res = res[:m.start()] + "[Invalid Citation]" + res[m.end():]
        logger.info(f"Finalized: {len(used)} citations."); return res, used

def main():
    try: s = RAGSearcher()
    except Exception as e: return print(f"Init error: {e}")
    n = 1
    while True:
        if not (q := input(f"\n### Question {n}: ").strip()): break
        res = s.retrieve(q); prc = CitationProcessor(res)
        txt, cits = prc.finalize(s.generate(q, prc.format_ctx()))
        print(f"\n{'='*50}\n**Answer:**\n{txt.strip()}")
        if cits:
            print("\n**Citations:**")
            for c in cits: print(f"• {c}")
        print("\n**Retrieved Evidence:**")
        for i, (cid, mt, dc) in enumerate(zip(prc.ids, prc.mts, prc.dcs), 1):
            ct = f"[{mt['doc_name']}, p.{mt['page']}, chunk:{cid}]"
            print(f"{i}. {ct}{' [USED]' if any(ct in c for c in cits) else ' [UNUSED]'}\n   \"{dc[:120].replace(chr(10), ' ')}...\"")
        print("="*50); n += 1

if __name__ == "__main__": main()