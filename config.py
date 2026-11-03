"""Settings shared by the ingestion script, the retriever and the Streamlit app."""
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

DATA_DIR = PROJECT_DIR / "data"
EXPIRED_DIR = PROJECT_DIR / "negative_examples"
CHUNKS_FILE = PROJECT_DIR / "index" / "chunks.jsonl"  # local copy of every chunk, used by BM25

INDEX_NAME = "hr-policies"
NAMESPACE = "northstar_demo"  # one namespace per tenant
TEXT_FIELD = "chunk_text"  # the field Pinecone embeds (set when the index was created)
EMBED_MODEL = "llama-text-embed-v2"

MAX_CHUNK_WORDS = 250  # sections longer than this are split
OVERLAP_WORDS = 40

# Retrieval
CANDIDATES_K = 20  # stage 1: candidates from each of dense and BM25 before fusion
RRF_K = 60  # standard Reciprocal Rank Fusion constant
RERANK_MODEL = "bge-reranker-v2-m3"
FINAL_K = 5  # stage 2: chunks passed to the LLM
# Generation
LLM_MODEL = "openai/gpt-oss-120b"  # served by Groq
MAX_ANSWER_TOKENS = 600
PEOPLE_CONTACT = "people@northstar-demo.example"

# Chunks scoring below this are dropped as irrelevant. Chosen with eval_retrieval.py: 0.01 keeps
# Recall@5 at 0.91 while 5 of 7 no-answer questions still retrieve nothing. Higher values cut
# the second document from multi-document questions. Access control does not depend on this.
MIN_RERANK_SCORE = 0.01
# Experimental: keep a below-threshold chunk if dense and BM25 both ranked it this high or better
# (targets outdated-premise questions such as Q40). 0 switches it off. See eval_retrieval.py --rescue.
RESCUE_FIRST_STAGE_RANK = 0
