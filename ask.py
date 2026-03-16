import chromadb, logging, ollama, os, re, time, requests
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

os.environ["TRANSFORMERS_OFFLINE"] = "1"
load_dotenv()
logging.basicConfig(filename="ask.log", filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class RAGSearcher:
    def __init__(self, db: str = os.getenv("DB_DIR",''), cl: str = os.getenv("COLLECTION_NAME",''), m: str = os.getenv("MODEL",'')):
        self.m, self.p = m, "Answer based ONLY on context. Cite as [chunk X]. If missing, say 'Not found.'"
        logger.info(f"Init: {m}")
        try: requests.get("http://127.0.0.1:11434/", timeout=2)
        except: logger.error("Ollama Offline")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=os.getenv("EMBEDDING_MODEL",''), device="cuda")
        self.coll = chromadb.PersistentClient(path=db).get_collection(cl, embedding_function=ef) # type: ignore

    def retrieve(self, q: str, n: int = 6):
        t = time.time()
        res = self.coll.query(query_texts=[q], n_results=n)
        logger.info(f"Retrieved {n} chunks in {time.time()-t:.4f}s")
        return dict(res)

    def generate(self, q: str, ctx: str):
        t = time.time()
        try:
            r = ollama.chat(model=self.m, messages=[{'role': 'system', 'content': self.p}, {'role': 'user', 'content': f"CONTEXT:\n{ctx}\n\nQ: {q}"}], options={'num_ctx': 8192, 'keep_alive': '10m'})
            d, ec, ed = time.time()-t, r.get('eval_count', 1), r.get('eval_duration', 1)/1e9
            tps = ec/ed
            if tps < 10: 
                logger.warning(f"THROTTLE: {tps:.2f} tps. Applying 3s cooldown."); time.sleep(3)
            logger.info(f"Ollama: {d:.2f}s | {tps:.2f} tps")
            return r['message']['content'], tps
        except Exception as e:
            logger.error(f"Ollama error: {e}"); return "Error: LLM unreachable.", 0

class CitationProcessor:
    def __init__(self, res: dict): self.ids, self.mts, self.dcs = res['ids'][0], res['metadatas'][0], res['documents'][0]
    def format_ctx(self): return '\n\n'.join([f"Context from [chunk {i}]:\n{d}" for i, d in zip(self.ids, self.dcs)])
    def finalize(self, raw: str):
        res, used = raw, []
        for m in sorted(re.finditer(r'\[?[Cc]hunk\s+(\d+)[^\]]*\]?', raw), key=lambda x: x.start(), reverse=True):
            cid = m.group(1)
            if cid in self.ids:
                mt = self.mts[self.ids.index(cid)]; ct = f"[{mt['doc_name']}, p.{mt['page']}, chunk:{cid}]"
                res = res[:m.start()] + ct + res[m.end():]
                if ct not in used: used.append(ct)
            else: 
                logger.warning(f"Hallucination: Chunk {cid} not in results."); res = res[:m.start()] + "[Citation Error]" + res[m.end():]
        return res, used

def main():
    try: s = RAGSearcher()
    except Exception as e: return print(f"Init error: {e}")
    n, last_tps = 1, 0
    while True:
        status = f" | Last: {last_tps:.1f} tps" if last_tps else ""
        if not (q := input(f"\n### Q{n}{status}: ").strip()): break
        res = s.retrieve(q); prc = CitationProcessor(res)
        raw, last_tps = s.generate(q, prc.format_ctx())
        txt, cits = prc.finalize(raw)
        print(f"\n{'='*50}\n**Answer:**\n{txt.strip()}\n\n**Citations:**")
        [print(f"• {c}") for c in cits] if cits else print("None found.")
        print(f"{'='*50}"); n += 1

if __name__ == "__main__": main()