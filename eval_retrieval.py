"""Retrieval evaluation on the golden set, with no LLM involved.

For questions with expected passages: Recall@k (share of expected chunks retrieved), MRR, and noise
(chunks sent to the LLM that are not expected evidence). For questions that should retrieve nothing
(access-denied, unanswerable, ambiguous): how often retrieval correctly comes back empty. For every
question: leaks of forbidden documents (restricted or expired files listed under negative_examples).
Selection uses retriever.select(), the same function the app uses.

    python eval_retrieval.py                         # stage-1 and stage-2 metrics + threshold sweep
    python eval_retrieval.py --from-raw              # re-score the saved candidates, no API calls
    python eval_retrieval.py --include-expired       # adversarial run (index must include expired docs)
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

import config
from ingest import slugify
from retriever import Hit, HybridRetriever, select

GOLDEN = config.PROJECT_DIR / "evaluation" / "golden_50.json"
OUT = config.PROJECT_DIR / "evaluation" / "retrieval_results.csv"
RAW = config.PROJECT_DIR / "evaluation" / "retrieval_raw.json"  # every candidate's scores, for offline analysis
EVAL_DATE = date(2026, 10, 18)  # fixed evaluation date from the README, for repeatable runs
THRESHOLDS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
RESCUE_RANKS = [0, 1, 2]


def chunk_id(source_uri: str, heading: str) -> str:
    return f"{Path(source_uri).stem}#{slugify(heading)}"


def metrics(expected: set[str], retrieved: list[str]) -> tuple[float, float]:
    recall = len(expected & set(retrieved)) / len(expected)
    rr = next((1 / i for i, cid in enumerate(retrieved, start=1) if cid in expected), 0.0)
    return recall, rr


def expected_ids(q: dict) -> set[str]:
    return {chunk_id(p["source_uri"], p["heading"]) for p in q["relevant_passages"]}


def forbidden_docs(q: dict) -> set[str]:
    return {Path(n["source_uri"]).stem for n in q["negative_examples"]}


def score(rows: list[tuple[dict, list[Hit]]], threshold: float, final_k: int, rescue: int) -> dict:
    """Run the app's selection rule over saved candidates and score it.

    Questions with no expected passages should come back empty; the rest are scored on recall, MRR and
    noise. Q14-style live-data questions point at a 'no live data' passage, so they count as answerable.
    """
    recalls, rrs, multi, noise, sent = [], [], [], [], []
    empty_ok, should_be_empty, leaks, lost = 0, 0, 0, []
    for q, cands in rows:
        final = [h.id for h in select(cands, final_k, threshold, rescue)[0]]
        leaks += any(cid.split("#")[0] in forbidden_docs(q) for cid in final)
        expected = expected_ids(q)
        if not expected:
            should_be_empty += 1
            empty_ok += not final
            continue
        r, rr = metrics(expected, final)
        recalls.append(r)
        rrs.append(rr)
        noise.append(len(set(final) - expected))
        sent.append(len(final))
        if q["category"] == "multi-document":
            multi.append(r)
        if r < 1:
            lost.append(q["id"])
    mean = lambda xs: sum(xs) / len(xs)
    return {"recall": mean(recalls), "mrr": mean(rrs), "multi_recall": mean(multi), "noise": mean(noise),
            "sent": mean(sent), "empty_ok": f"{empty_ok}/{should_be_empty}", "leaks": leaks, "lost": lost}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-expired", action="store_true", help="switch off the status/date filter")
    parser.add_argument("--from-raw", action="store_true", help=f"re-score {RAW.name} instead of calling Pinecone")
    args = parser.parse_args()

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if args.from_raw:
        raw = json.loads(RAW.read_text(encoding="utf-8"))
        rows = [(q, [Hit(**h) for h in raw[q["id"]]]) for q in golden]
    else:
        retriever = HybridRetriever()
        rows = []
        for q in golden:
            res = retriever.search(q["question"], q["user_roles"], EVAL_DATE, args.include_expired,
                                   min_rerank_score=0.0, final_k=config.CANDIDATES_K * 2, rescue_rank=0)
            rows.append((q, res.candidates))  # candidates are sorted by rerank score
            print(f"{q['id']} done", end="\r")
        RAW.write_text(json.dumps({q["id"]: [vars(h) | {"meta": None} for h in c] for q, c in rows}, indent=1),
                       encoding="utf-8")

    stage1 = [metrics(expected_ids(q), [h.id for h in sorted(c, key=lambda h: h.rrf_score, reverse=True)])
              for q, c in rows if q["relevant_passages"]]
    print(f"\nStage 1 (hybrid, fused top-{config.CANDIDATES_K}): "
          f"Recall@{config.CANDIDATES_K} = {sum(r for r, _ in stage1) / len(stage1):.3f} over {len(stage1)} questions")

    header = (f"{'threshold':>9} {'rescue':>6} {'Recall@5':>9} {'multi-doc':>9} {'MRR':>6} {'sent':>5} {'noise':>6} "
              f"{'empty when expected':>20} {'leaks':>6}  questions missing evidence")
    print(f"\nStage 2 (reranked, max {config.FINAL_K}). sent/noise = chunks sent / irrelevant chunks sent, "
          f"per answerable question.\n{header}")
    for rescue in RESCUE_RANKS:
        for t in THRESHOLDS:
            s = score(rows, t, config.FINAL_K, rescue)
            mark = " <- current" if (t, rescue) == (config.MIN_RERANK_SCORE, config.RESCUE_FIRST_STAGE_RANK) else ""
            print(f"{t:>9} {rescue:>6} {s['recall']:>9.3f} {s['multi_recall']:>9.3f} {s['mrr']:>6.3f} {s['sent']:>5.2f} "
                  f"{s['noise']:>6.2f} {s['empty_ok']:>20} {s['leaks']:>6}  {','.join(s['lost'])}{mark}")
        print()

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "user_roles", "expected_chunks", "retrieved_chunks_with_rerank_score",
                    "recall_at_5", "reciprocal_rank", "forbidden_leak", "outcome"])
        for q, cands in rows:
            final, outcome = select(cands, config.FINAL_K, config.MIN_RERANK_SCORE, config.RESCUE_FIRST_STAGE_RANK)
            expected = expected_ids(q)
            r, rr = metrics(expected, [h.id for h in final]) if expected else ("", "")
            w.writerow([q["id"], q["category"], ";".join(q["user_roles"]), ";".join(sorted(expected)),
                        ";".join(f"{h.id}={h.rerank_score:.3f}" for h in final), r, rr,
                        any(h.id.split("#")[0] in forbidden_docs(q) for h in final), outcome])
    print(f"Per-question results at threshold {config.MIN_RERANK_SCORE}, rescue {config.RESCUE_FIRST_STAGE_RANK}: "
          f"{OUT.relative_to(config.PROJECT_DIR)}")


if __name__ == "__main__":
    main()
