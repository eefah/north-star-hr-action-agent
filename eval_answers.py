"""End-to-end answer evaluation on the golden set: run the bot, then have a judge LLM score each answer.

Retrieval is scored without an LLM (all / some / none of the expected chunks reached the prompt).
Answer quality, citations and faithfulness are scored by a judge model from a different family than
the generator, so the bot is not marking its own work. Rubric: evaluation/README in the pack.

    python eval_answers.py                     # run the bot on all 50 questions, then judge
    python eval_answers.py --rejudge           # judge the saved answers again, no new bot calls
    python eval_answers.py --ids Q06,Q40       # a subset
    python eval_answers.py --include-expired   # stale-policy run (index must include expired docs)
"""
import argparse
import csv
import json
import time
from pathlib import Path

from groq import Groq

import config
from eval_retrieval import EVAL_DATE, GOLDEN, chunk_id
from rag import RAG, format_context

JUDGE_MODEL = "qwen/qwen3.8-27b"
EVAL_DIR = config.PROJECT_DIR / "evaluation"

JUDGE_PROMPT = """You are grading an HR policy assistant. You see the user's question, the policy excerpts the assistant was given, the assistant's answer, and a reference answer written by the policy author.

Grade with this rubric:
- answer_score: 2 = correct, complete and grounded, or an appropriate clarifying question / honest "I can't find this" when that is what the reference expects; 1 = partly correct or incomplete, with no invented material fact; 0 = wrong, invented, misses a critical exception, or refuses when the excerpts contain the answer.
- citation_score: 2 = every substantive policy claim cites an excerpt ID that supports it; 1 = some claims uncited or cited to an excerpt that does not support them; 0 = citations missing or wrong. null when the answer makes no policy claim (a pure clarifying question or refusal).
- faithful: true if every factual claim in the answer is supported by the excerpts (an answer with no factual claims is faithful). Judge against the EXCERPTS, not the reference: a correct fact that is not in the excerpts is unfaithful.
- unsupported_claims: the claims not supported by the excerpts, quoted briefly. Empty list if none.
- failure_type: "none", or the main cause of a lost point, one of: "retrieval" (needed evidence missing from the excerpts), "stale_metadata" (used an expired policy), "missing_multi_doc" (only part of a multi-document answer retrieved), "generation" (evidence was there but the answer misused or ignored it), "ambiguous_intent" (should have asked a clarifying question, or asked one needlessly), "absent_information" (corpus lacks the answer and the assistant failed to say so).
- rationale: one or two sentences.

Return only a JSON object with exactly these keys: answer_score, citation_score, faithful, unsupported_claims, failure_type, rationale."""

JUDGE_KEYS = {"answer_score", "citation_score", "faithful", "unsupported_claims", "failure_type", "rationale"}

COLUMNS = ["id", "question", "category", "user_roles", "actual_answer", "retrieved_sources", "retrieved_chunk_ids",
           "retrieval_score_0_to_2", "answer_score_0_to_2", "citation_score_0_to_2", "faithful", "forbidden_leak",
           "latency_seconds", "failure_analysis", "suggested_fix", "run_status"]


def retrieval_score(expected: set[str], retrieved: list[str]) -> int | str:
    """2 = all expected chunks retrieved, 1 = some, 0 = none. N/A when the question has no required evidence."""
    if not expected:
        return "N/A"
    found = len(expected & set(retrieved))
    return 2 if found == len(expected) else 1 if found else 0


def run_bot(golden: list[dict], include_expired: bool, min_rerank_score: float) -> list[dict]:
    rag = RAG()
    rag.llm = Groq(max_retries=8)
    runs = []
    for q in golden:
        a = rag.answer(q["question"], q["user_roles"], EVAL_DATE, include_expired, min_rerank_score=min_rerank_score)
        runs.append({"id": q["id"], "answer": a.text, "citations": a.citations, "invalid_citations": a.invalid_citations,
                     "hits": [{"id": h.id, "source_uri": h.meta["source_uri"], "rerank": h.rerank_score} for h in a.hits],
                     "context": format_context(a.hits), "latency_s": a.latency_s, "model": a.model,
                     "outcome": a.retrieval.outcome})
        print(f"bot   {q['id']}  {sum(a.latency_s.values()):5.2f} s  {len(a.hits)} chunks  {a.retrieval.outcome}",
              flush=True)
    return runs


def judge(llm: Groq, q: dict, run: dict) -> dict:
    if not run["hits"]:  # refusal gate fired: no excerpts, so the answer is the fixed refusal text
        excerpts = "(none: retrieval returned nothing, so the assistant gave its fixed refusal)"
    else:
        excerpts = run["context"]
    message = (f"Question category: {q['category']}\nUser roles: {', '.join(q['user_roles'])}\n\n"
               f"Question: {q['question']}\n\nReference answer: {q['expected_answer']}\n\n"
               f"Excerpts given to the assistant:\n{excerpts}\n\nAssistant's answer:\n{run['answer']}")
    for _ in range(3):  # the judge occasionally returns a verdict with a key missing; ask again
        response = llm.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": message}],
            temperature=0,
            max_completion_tokens=400,  # free tier allows 1,000 output tokens a minute for this model
            response_format={"type": "json_object"},
            reasoning_effort="none",
        )
        verdict = json.loads(response.choices[0].message.content)
        if JUDGE_KEYS <= verdict.keys():
            return verdict
    raise ValueError(f"{q['id']}: judge verdict missing {JUDGE_KEYS - verdict.keys()}: {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rejudge", action="store_true", help="reuse saved answers, only run the judge")
    parser.add_argument("--ids", help="comma-separated question IDs, e.g. Q06,Q40")
    parser.add_argument("--include-expired", action="store_true", help="switch off the status/date filter")
    parser.add_argument("--label", help="suffix for the output files, e.g. stale_filtered")
    parser.add_argument("--min-rerank-score", type=float, default=config.MIN_RERANK_SCORE,
                        help="override the threshold for a calibration run (use with --label)")
    args = parser.parse_args()

    suffix = f"_{args.label}" if args.label else "_expired" if args.include_expired else ""
    raw_file = EVAL_DIR / f"answers_raw{suffix}.json"
    out_file = EVAL_DIR / f"results{suffix}.csv"
    summary_file = EVAL_DIR / f"answer_summary{suffix}.json"

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if args.ids:
        wanted = set(args.ids.split(","))
        golden = [q for q in golden if q["id"] in wanted]

    saved = json.loads(raw_file.read_text(encoding="utf-8")) if raw_file.exists() else {}
    if not args.rejudge:
        for run in run_bot(golden, args.include_expired, args.min_rerank_score):
            saved[run["id"]] = run
        raw_file.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")

    llm = Groq(max_retries=8)  # free-tier rate limits: the SDK waits and retries on 429
    rows = []
    for q in golden:
        run = saved[q["id"]]
        verdict = judge(llm, q, run)
        run["judge"] = verdict
        expected = {chunk_id(p["source_uri"], p["heading"]) for p in q["relevant_passages"]}
        retrieved = [h["id"] for h in run["hits"]]
        forbidden = {Path(n["source_uri"]).stem for n in q["negative_examples"]}
        citation = verdict.get("citation_score")
        rows.append({
            "id": q["id"], "question": q["question"], "category": q["category"],
            "user_roles": ";".join(q["user_roles"]), "actual_answer": run["answer"],
            "retrieved_sources": ";".join(dict.fromkeys(h["source_uri"] for h in run["hits"])),
            "retrieved_chunk_ids": ";".join(retrieved),
            "retrieval_score_0_to_2": retrieval_score(expected, retrieved),
            "answer_score_0_to_2": verdict["answer_score"],
            "citation_score_0_to_2": "N/A" if citation is None else citation,
            "faithful": verdict["faithful"],
            "forbidden_leak": any(cid.split("#")[0] in forbidden for cid in retrieved),
            "latency_seconds": round(sum(run["latency_s"].values()), 2),
            "failure_analysis": "" if verdict["failure_type"] == "none"
                                else f"{verdict['failure_type']}: {verdict['rationale']}",
            "suggested_fix": "",
            "run_status": f"RUN {EVAL_DATE} {config.LLM_MODEL}; judge {JUDGE_MODEL}",
        })
        print(f"judge {q['id']}  answer {verdict['answer_score']}  citation {citation}  "
              f"faithful {verdict['faithful']}  {verdict['failure_type']}", flush=True)
        time.sleep(1)
    raw_file.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")

    with out_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    def mean(key: str) -> dict:
        vals = [r[key] for r in rows if r[key] != "N/A"]
        return {"mean": round(sum(vals) / len(vals), 3) if vals else None, "n": len(vals)}

    latencies = sorted(r["latency_seconds"] for r in rows)
    summary = {
        "questions": len(rows),
        "retrieval": mean("retrieval_score_0_to_2"),
        "answer": mean("answer_score_0_to_2"),
        "citation": mean("citation_score_0_to_2"),
        "faithfulness": round(sum(r["faithful"] is True for r in rows) / len(rows), 3),
        "answer_fully_correct": round(sum(r["answer_score_0_to_2"] == 2 for r in rows) / len(rows), 3),
        "forbidden_leaks": sum(r["forbidden_leak"] for r in rows),
        "latency_s": {"median": latencies[len(latencies) // 2], "p90": latencies[int(len(latencies) * 0.9)],
                      "max": latencies[-1]},
        "by_category": {},
        "settings": {"eval_date": str(EVAL_DATE), "generator": config.LLM_MODEL, "judge": JUDGE_MODEL,
                     "final_k": config.FINAL_K, "candidates_k": config.CANDIDATES_K,
                     "min_rerank_score": args.min_rerank_score,
                     "rescue_first_stage_rank": config.RESCUE_FIRST_STAGE_RANK, "include_expired": args.include_expired},
    }
    for cat in dict.fromkeys(r["category"] for r in rows):
        group = [r for r in rows if r["category"] == cat]
        summary["by_category"][cat] = {
            "n": len(group),
            "answer_mean": round(sum(r["answer_score_0_to_2"] for r in group) / len(group), 2),
            "faithful": sum(r["faithful"] is True for r in group),
        }
    summary_file.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"\nPer-question results: {out_file.relative_to(config.PROJECT_DIR)}")


if __name__ == "__main__":
    main()
