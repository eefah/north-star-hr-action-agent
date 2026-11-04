# Northstar HR Policy Assistant

A RAG chatbot that answers HR policy questions from 10 policy documents, cites a policy section for every claim, and says "I can't find this" when the policies don't cover the question.

Built for Week 2 of Mastering Agentic AI (use case 1: Enterprise Policy Q&A Bot). The policies describe Northstar Studio, a fictional UK employer. They are synthetic and are not real HR or legal guidance.

## How it works

1. **Access filter.** The user's role, the tenant and the policy date become a metadata filter, so search only sees current policies that this role is allowed to read.
2. **Hybrid retrieval.** Pinecone dense search (`llama-text-embed-v2`) and local BM25 each return their top 20. Reciprocal Rank Fusion merges the two lists.
3. **Rerank.** `bge-reranker-v2-m3` keeps the top 5 chunks with a score of at least 0.01.
4. **Refusal gate.** If no chunk survives reranking, the app returns a fixed refusal and never calls the LLM.
5. **Generation.** `openai/gpt-oss-120b` on Groq answers from the excerpts only and cites chunk IDs such as `[expenses#travel-and-meals]`.
6. **Citation check.** Any cited ID that wasn't actually retrieved is flagged in the UI.

| File | Purpose |
| --- | --- |
| `config.py` | All settings: models, top-k, threshold, index name |
| `ingest.py` | Parse policy headers into metadata, chunk by section, upload to Pinecone |
| `retriever.py` | Filtered hybrid search, RRF fusion, reranking |
| `rag.py` | Refusal gate, prompt, generation, citation check |
| `app.py` | Streamlit chat UI with a role switcher and retrieval trace |
| `eval_retrieval.py` | Recall@5, MRR and leak checks on the 50-question golden set |
| `eval_answers.py` | End-to-end answer evaluation with an LLM judge (`qwen/qwen3.8-27b`) |
| `evaluation/` | Golden sets, results and the evaluation report |
| `techstack.md` | Tools, data flow and security posture for enterprise review |
| `DATA_PACK.md` | The original guide that came with the synthetic policy pack |

## Setup

Requires Python 3.12, a [Groq](https://console.groq.com) API key and a [Pinecone](https://app.pinecone.io) API key. The free tiers of both are enough.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```
GROQ_API_KEY=...
PINECONE_API_KEY=...
```

Create the Pinecone index once, with integrated embedding:

```python
from pinecone import Pinecone
Pinecone().create_index_for_model(
    name="hr-policies", cloud="aws", region="us-east-1",
    embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
)
```

Then ingest the policies and start the app:

```bash
python ingest.py          # 10 documents -> 51 chunks
streamlit run app.py
```

## Evaluation

```bash
python eval_retrieval.py   # retrieval metrics and threshold sweep, no LLM calls
python eval_answers.py     # bot answers + LLM-judge scores -> evaluation/results.csv
```

Both scripts use a fixed evaluation date of 18 October 2026, so results are repeatable. The full report is in `evaluation/report.md`.

To run the stale-policy test, index the expired expenses policy alongside the current one. Then ask the outdated-premise questions twice: once with the status and date filter on (the expired policy should never be retrieved), and once with it off (showing what the filter prevents):

```bash
python ingest.py --include-expired
python eval_answers.py --ids Q06,Q21,Q40 --label stale_filtered
python eval_answers.py --ids Q06,Q21,Q40 --include-expired --label stale_unfiltered
python ingest.py          # restore the baseline index afterwards
```

## Week 3: HR Policy Action Agent (supervised demo, synthetic data only)

The agent (`agent/`) runs annual leave requests and policy questions as a LangGraph workflow: it checks
eligibility and dates against verified policy text, prepares a draft, and saves it locally only after the
synthetic employee approves that exact draft. Nothing is submitted to PeopleHub or sent to anyone.

```bash
pip install -r requirements.txt
streamlit run agent_app.py                      # the demonstration interface
python -m unittest discover tests               # offline tests (fake Pinecone and Groq)
python -m unittest -v tests.live_evidence       # live checks against Pinecone and Groq (needs .env)
python -m agent.demo --delete-all               # delete all demo checkpoints, drafts and logs
```

Demo data (checkpoints, approved drafts, diagnostic logs) is kept in `demo_data/`, which is Git-ignored and
owner-only. This is a demonstration, not production-grade privacy compliance: use synthetic data only.
