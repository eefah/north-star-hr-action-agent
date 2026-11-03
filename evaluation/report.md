# Evaluation report

On a 50-question golden set, every answer was faithful to the retrieved policies, 40 of 50 answers were fully correct, and no restricted or expired document reached a user who shouldn't see it. The main weaknesses are answers that drop secondary conditions, and questions that quote outdated numbers.

## Setup

| Item | Value |
| --- | --- |
| Run date / evaluation date | 25 Oct 2026 / policies in force on 18 Oct 2026 |
| Corpus | Northstar synthetic demo 2.0: 10 policies, 51 chunks (plus 1 expired policy for the stale test) |
| Parsing and chunking | One chunk per `##` section; 250-word windows with 40-word overlap only for longer sections (none were longer; max 66 words); title and heading prepended; word counts, not tokens |
| Embedding / vector store | `llama-text-embed-v2` (1024-d, cosine), Pinecone serverless, AWS us-east-1 |
| Retriever | Dense top 20 + BM25 top 20, RRF (k = 60), `bge-reranker-v2-m3`, top 5 with rerank score >= 0.01 |
| Filters | tenant, role overlap, `status = current`, effective dates covering the evaluation date |
| Generation | `openai/gpt-oss-120b` on Groq, temperature 0, max 600 tokens, system prompt in `rag.py` |
| Judge | `qwen/qwen3.8-27b` on Groq, temperature 0, prompt in `eval_answers.py` |
| Ground truth | Author-written synthetic references; not reviewed by an HR subject matter expert |
| Retrieval audit runs | 3 Nov 2026, same evaluation date, corpus and models. Retrieval sweep over thresholds 0 to 0.1 and rescue settings; full answer runs at thresholds 0.01 and 0.002; rescue off. The judge now retries a verdict with missing fields |

Scripts: `eval_retrieval.py` (retrieval only, no LLM) and `eval_answers.py` (end-to-end with the judge). Per-question results are in `results.csv`, and raw answers with judge verdicts are in `answers_raw.json`. The 3 Nov answer runs are in `results_cal_t0010.csv` (threshold 0.01) and `results_cal_t0002.csv` (threshold 0.002), with matching `answer_summary_cal_*.json` and `answers_raw_cal_*.json`.

## Results

| Metric | Result | Denominator |
| --- | --- | --- |
| Faithfulness (every claim supported by retrieved excerpts) | **100%** | 50 |
| Answers fully correct (score 2) | **80%** (40) | 50 |
| Answer score, mean | 1.78 / 2 | 50 |
| Citation score, mean | 2.00 / 2 | 42 (8 N/A: refusals and clarifying questions) |
| Retrieval score, mean | 1.81 / 2 | 43 (7 N/A: no required evidence) |
| Recall@5 (expected chunks reaching the prompt) | 0.91 | 43 |
| MRR | 0.95 | 43 |
| Forbidden documents retrieved | **0** | 50 |
| Latency, median / p90 / max | 4.2 s / 7.6 s / 9.1 s | 50 |

Answer score distribution: 40 scored 2, 9 scored 1 and 1 scored 0.

| Category | Fully correct |
| --- | --- |
| Direct | 13 / 16 |
| Exception | 9 / 10 |
| Multi-document | 4 / 7 |
| Access-allowed | 4 / 5 |
| Access-denied | 3 / 3 |
| Keyword | 2 / 2 |
| Ambiguous (clarifying question expected) | 2 / 2 |
| Unanswerable, missing policy | 2 / 2 |
| Unanswerable, live data | 1 / 1 |
| Outdated premise | 0 / 2 |

All 8 questions that should get a refusal or a clarifying question got one: 3 access-denied, 2 ambiguous, 2 missing policy and 1 live data.

**Latency.** Retrieval took a median of 0.6 s. Generation took about 0.5 s for the first 10 questions, then about 3.5 s for the rest of the run. This pattern looks like Groq free-tier throttling under back-to-back requests, but I haven't confirmed it. Single questions in the app return in 1 to 3 s. The 5-second target is met at the median but missed at p90 under evaluation load.

### Stale-policy challenge

I indexed the expired expenses policy (60-day deadline, £40 dinner cap) alongside the current one (30 days, £30). Then I asked the three questions that touch it, with the filter on and with it off.

| Question | Filter on: expired doc retrieved? | Filter off: expired doc retrieved? | Filter off: stale rule presented as current? |
| --- | --- | --- | --- |
| Q06 "60 days and £40?" | No | Yes | No. It said the 60 days came from an expired version |
| Q21 "deadline for expenses?" | No | Yes | No. It answered 30 days from the current policy |
| Q40 "60 days and £40, is that right?" | No | Yes | No. It said these were the expired rules, replaced on 1 Oct 2026 |

The metadata filter is the real control: with it on, the expired policy never reaches the prompt. With it off, the model still didn't pass off stale rules as current. Each excerpt carries `status="expired"`, and the model used that label. This is a useful second layer, but it isn't something to rely on.

### Retrieval relevance audit (3 Nov 2026)

**Question.** In the Q01 trace, one chunk scored 0.997 with the reranker and four scored between 0.022 and 0.265, yet all five reached the prompt. Is the system filling slots rather than selecting evidence?

**How selection works.** Candidates are sorted by reranker score, not by RRF. At most 5 are kept, and any chunk below 0.01 is dropped. Five is a ceiling: 26 of 50 golden questions send fewer than 5 chunks, and 6 send none. Q01 sent five because five chunks cleared 0.01. `onboarding#first-month` was left out even though its RRF score (0.0303) beat two chunks that were sent, because its reranker score (0.018) was lower. Reranker scores rank chunks for one question. They aren't calibrated probabilities and can't be compared reliably across questions.

**Why not raise the threshold.** Relevant and irrelevant chunks overlap heavily in reranker score. Across all 50 questions (top 20 candidates each, 58 relevant chunks, 942 others):

| Threshold | Relevant chunks kept (of 58) | Irrelevant chunks above threshold (of 942) |
| --- | --- | --- |
| 0.002 | 56 | 145 |
| 0.005 | 51 | 95 |
| **0.01 (current)** | **51** | **62** |
| 0.02 | 47 | 35 |
| 0.05 | 45 | 16 |
| 0.1 | 40 | 11 |

Removing Q01's extra chunks would need a threshold near 0.3. That would drop needed second-document evidence in Q04 (0.026), Q07 (0.018), Q09 (0.014), Q36 (0.08) and Q46 (0.014). A cutoff relative to the top score fails for the same reason: Q09's needed chunk scores 1.4% of its top chunk. Q01's extra chunks cost tokens and clutter the Sources panel, but they didn't affect the answer, which scored 2 and cited only the correct chunk.

**Threshold sweep (retrieval only, live run, 3 Nov 2026).** *Noise* is the number of irrelevant chunks sent per answerable question.

| Threshold | Recall@5 | Multi-doc recall | Chunks sent | Noise | Correct empty retrieval | Leaks |
| --- | --- | --- | --- | --- | --- | --- |
| 0.002 | 0.948 | 0.893 | 3.35 | 2.09 | 5/7 | 0 |
| 0.005 | 0.913 | 0.821 | 2.72 | 1.53 | 5/7 | 0 |
| **0.01 (current)** | **0.913** | **0.821** | **2.23** | **1.05** | **5/7** | **0** |
| 0.02 | 0.864 | 0.786 | 1.77 | 0.67 | 5/7 | 0 |
| 0.05 | 0.841 | 0.643 | 1.33 | 0.28 | 6/7 | 0 |
| 0.1 | 0.760 | 0.571 | 1.14 | 0.21 | 6/7 | 0 |

The live run reproduced the 25 Oct Recall@5 of 0.913. 0.005 is strictly worse than 0.01: same recall, more noise. Thresholds of 0.02 and above lose needed evidence (Q03, Q07, Q46 first). 0.002 is the only competitor. It recovers the second chunk for Q05 and Q06 with no change to refusals, but it doubles the noise. The answer-quality comparison below decides between them.

**Rescue experiment (rejected).** The rule kept a below-threshold chunk if dense and BM25 both ranked it in their top *n*. At *n* = 1 with threshold 0.01, Recall@5 rose from 0.913 to 0.924 (Q40 gets one of its two chunks). But Q45, an access-denied question, also kept a chunk (`sickness_absence#return-to-work`, reranker score 0.000) and stopped refusing at the gate. At *n* = 2, three questions that should refuse kept chunks (Q15, Q42, Q45). No document was leaked in either case, because rescued chunks still pass the permission filter. Rescue is switched off (`RESCUE_FIRST_STAGE_RANK = 0`).

**Refusal labels.** Every refusal now records why retrieval came back empty. These labels are internal only: the user always sees the same message, so a refusal never reveals whether a restricted document exists.

| Outcome | Meaning | Golden questions |
| --- | --- | --- |
| `no_candidates` | Nothing found in the documents this user may see | none (dense search always returns something) |
| `below_threshold` | Candidates found, all below the threshold | Q15, Q42 (missing policy), Q43, Q49 (access denied) |
| `check_retrieval` | Below threshold, but dense and BM25 both ranked one chunk 1st | Q40 (a real reranker miss), Q45 (access denied: a false alarm) |

`check_retrieval` is a hint for debugging, not proof. It caught the one real retrieval failure, with one false alarm.

**Transparency and safety changes.** The retrieval trace in the app and the CLI now shows the query, the threshold, the cap and the outcome. For every candidate it shows dense and BM25 rank, RRF score, reranker score, whether it passed the threshold, whether it was sent, and why: *sent*, *below threshold*, *beyond the 5-chunk cap* or *rescued*. `HybridRetriever.search()` now re-applies the permission filter locally to Pinecone's results, so a misconfigured index can't pass a restricted chunk to the reranker or the LLM. `eval_retrieval.py` scores with the same `select()` function as the app, and `--from-raw` re-scores saved candidates without API calls. 17 offline regression tests (`python -m unittest discover tests`) cover selection order, cap and threshold reasons, rescue, the refusal gate, role, tenant and date filtering, and the Q37 and Q38 citation cases.

**Answer quality at 0.01 vs 0.002 (same-day runs, all 50 questions, one run each).**

| Metric | 0.01 | 0.002 |
| --- | --- | --- |
| Answers fully correct | 78% (39) | 78% (39) |
| Answer score, mean | 1.76 | 1.76 |
| Multi-document answer mean | 1.57 | **1.71** |
| Outdated-premise answer mean | 0.5 | **2.0** |
| Retrieval score, mean | 1.81 | **1.88** |
| Faithfulness | 100% | 100% |
| Citation score, mean | 2.00 (41) | 2.00 (42) |
| Invalid citations flagged | 1 (Q37) | 0 |
| Forbidden leaks | 0 | 0 |
| Questions that should refuse, refused | 5 / 5 | 5 / 5 |
| Chunks sent per question | 2.08 | 3.08 |
| Latency, median / p90 | 4.0 s / 6.7 s | 3.4 s / 7.0 s |

The 0.01 run is a same-day repeat of the 25 Oct setup. It scored 39 fully correct against 40 on 25 Oct: Q07 rose to 2, while Q21 and Q48 fell to 1 with the same evidence. This is the run-to-run variation to expect from one generator pass and one judge pass.

Totals tie at 0.002, but the changed questions differ in kind:

| ID | 0.01 | 0.002 | Evidence changed? | Reading |
| --- | --- | --- | --- | --- |
| Q05 | 1 | 2 | Yes: gained `expenses#eligibility-and-submission` | Real gain: missing deadline chunk now retrieved |
| Q06 | 1 | 2 | Yes: same chunk | Real gain: both outdated numbers corrected |
| Q40 | 0 | 2 | Yes: same chunk | Real gain: refusal replaced by a correct, cited answer |
| Q48 | 1 | 2 | Yes: one irrelevant chunk added | Probably variation |
| Q07 | 2 | 1 | No | Variation: identical evidence |
| Q32 | 2 | 1 | Yes: three irrelevant chunks added | Judge variation: the two answers are nearly word for word the same |
| Q44 | 2 | 0 | Yes: gained `pay_and_salary_bands#confidentiality` | Judge error: identical answer, marked down because the extra excerpt says band ranges are confidential, but the asker is a People admin who may see them |
| Q27 | 2 | 1 | Yes: three irrelevant chunks added | Possible real regression: told a 3-day-a-week employee to plan for 2 office days, applying the full-time rule |

*Decision rule, set before the runs:* zero leaks and correct refusals are required; then prefer the higher multi-document answer score; a difference of one or two questions is within judge noise (see Judge variance), so on its own it doesn't justify more noise in the prompt.

**Decision.** Recommended: lower `MIN_RERANK_SCORE` from 0.01 to 0.002. *Status: not yet applied; the app still runs at 0.01.* Against the decision rule set before the runs: leaks stay at zero, every question that should refuse still refuses, and the multi-document answer score rises. Three gains (Q05, Q06, Q40) trace directly to evidence that was missing at 0.01. At most one loss (Q27) might be caused by the extra chunks; the others are judge or run-to-run variation. The cost is about one extra chunk per question in the prompt, plus more uncited items in the Sources panel. Q09's fourth chunk is still missed at 0.002. Before adopting, repeat both runs once more, or hand-check Q27, to confirm the gains hold beyond a single pass.

## Failure analysis

In the 25 Oct baseline, 10 answers scored below 2. None contained an invented fact. The last column shows the same question in the two 3 Nov runs. Q21 and Q48 also fell to 1 in the 3 Nov 0.01 run, with unchanged evidence (see Judge variance).

| ID | Expected evidence | Retrieved | What went wrong | Cause (from the trace) | Rerun 3 Nov: 0.01 / 0.002 |
| --- | --- | --- | --- | --- | --- |
| Q02 | carry-over | carry-over | Gave the 5-day limit, but left out the 15 Dec approval and 31 Mar use-by dates | Generation: evidence present, conditions dropped | 1 / 1 |
| Q07 | partner-benefit, eligibility | both | Left out "one block within 6 months" | Generation | 2 / 1 (same evidence: variation) |
| Q34 | before-day-one | before-day-one | Left out "at least 5 working days before start" | Generation | 1 / 1 |
| Q38 | how-to-request | how-to-request | Left out the expected date and leave dates to provide | Generation | 1 / 1 |
| Q50 | appeals | appeals | Left out who hears the appeal | Generation | 1 / 1 |
| Q35 | first-month, equipment | both | Left out "hybrid only after the first 4 weeks" | Generation | 1 / 1 |
| Q05 | eligibility-and-submission, equipment | equipment only | Left out the 30-day deadline and itemised receipt | Retrieval: needed chunk reranked to 0.005, below the 0.01 threshold | 1 / **2** |
| Q09 | 4 chunks incl. sickness evidence | 3 of 4 | Left out the self-certification and fit-note rules | Retrieval: evidence chunk ranked 7th (score 0.002) | 1 / 1 (4th chunk still missed) |
| Q06 | eligibility-and-submission, travel-and-meals | travel-and-meals only | Corrected the £40 to £30, but said the deadline wasn't covered | Retrieval: deadline chunk scored 0.003 | 1 / **2** |
| Q40 | same as Q06 | nothing | Refused entirely | Retrieval: dense and BM25 both ranked the right chunk 1st, but the reranker scored it 0.003, so the refusal gate fired | 0 / **2** |

**Pattern 1: concise answers drop conditions (6 questions).** In every case the evidence was in the prompt. Rule 10 of the system prompt ("answer first in one or two sentences") pushes the model to stop after the headline fact. *Change to test:* reword rule 10 to "always include the conditions, deadlines and exceptions that apply". *Rerun:* not yet run.

**Pattern 2: the reranker penalises questions that contradict the policy (Q06, Q40).** A question quoting outdated numbers ("60 days, £40") is a poor match for the chunk that says "30 days, £30". The cross-encoder scores that pair as irrelevant, even though both first-stage retrievers ranked it first. *Change to test:* keep a chunk that both dense and BM25 rank first, even when it falls below the rerank threshold. Alternatively, rewrite the query to strip the user's numbers before reranking. *Rerun:* the rescue rule was tested on retrieval on 3 Nov 2026 and rejected, because it also stopped an access-denied question from refusing. A threshold of 0.002 fixed both: Q06 and Q40 scored 2 (see Retrieval relevance audit). Query rewriting is not yet run.

**Pattern 3: multi-document recall (Q05, Q09).** The second policy's chunk ranks just below the threshold. *Change to test:* lower the threshold to 0.002 for chunks from a document that is already retrieved, or decompose multi-part questions. *Rerun:* a global threshold of 0.002 was tested on 3 Nov 2026. Q05 rose to 2; Q09's fourth chunk is still missed (see Retrieval relevance audit).

### Issues the judge missed

- **Malformed citations.** In Q38 the model put a zero-width character inside the citation (`[ parent​al_leave#how-to-request ]`). In Q37 it dropped the document prefix (`[partner-benefit]`). The app's citation check didn't parse either one, so neither was flagged, yet the judge gave both answers full citation marks. *Fix:* strip Unicode spacing characters before parsing, and flag any bracketed token that isn't a retrieved ID. *Fixed 3 Nov 2026:* `rag.py` now strips zero-width characters, accepts padded brackets and reports `[partner-benefit]`-style tokens as invalid citations. Regression tests cover both cases.
- **Judge variance.** Q21 scored 2 in the baseline run and 1 in the stale-test runs, for almost identical answers. It scored 1 again in both 3 Nov runs. Q32 also scored 2 and 1 for near-identical answers on 3 Nov. A single judge pass has roughly one point of noise on borderline answers.
- **Judge penalised permitted content (Q44, 3 Nov, threshold 0.002).** The judge gave 0 to a correct salary-band answer for a People admin, because an extra excerpt says band ranges are confidential. The judge prompt shows the user's roles but doesn't say what each role may see. *Fix:* add role permissions to the judge prompt.
- **Incomplete verdict.** On 3 Nov the judge returned a verdict with the `faithful` field missing (Q27), which stopped the run. `eval_answers.py` now asks again, up to three times.
- **Retrieval scored 0 but answered correctly (Q14).** The expected passage ("live balances are not held here") wasn't retrieved. The model still refused correctly, because rule 7 of the prompt covers personal data.

## Limitations

- The policies are synthetic, and the references haven't been reviewed by an HR subject matter expert.
- The corpus has a single tenant, so tenant isolation isn't demonstrated.
- Roles are picked in the UI, not taken from an authenticated login.
- The LLM judge is a single pass with no human calibration. Judge scores should be read as approximate (see Judge variance above).
- Latency was measured on Groq's free tier.
- The threshold was calibrated on 58 relevant chunks, and the Q09 margin between keep and drop is about 0.01. Re-check it whenever the corpus, chunking or reranker changes.
- With 51 short chunks, dense search alone reaches Recall@5 of 1.00, so this corpus is too small to show the benefit of hybrid retrieval.

## Conclusion and next steps

The system is faithful and safe on this corpus. It never invented a fact, never leaked a restricted or expired document, and refused or asked for clarification whenever it should. Its answers are sometimes incomplete. Next steps, in order of expected impact:

1. Adopt the 0.002 threshold after one confirming rerun (see Retrieval relevance audit). This fixes Q05, Q06 and Q40.
2. Reword prompt rule 10 to keep conditions and exceptions, then rerun all 50 questions. This is still the largest source of lost points (Q02, Q34, Q35, Q38, Q50 in both 3 Nov runs).
3. Add more multi-document questions to the golden set, so the threshold is calibrated on more than 7 of them.
4. Calibrate the judge against a hand-scored sample of 10 answers, and give it the role permissions (Q44).
5. If outdated-premise questions still fail on a larger set, test rewriting the query to strip the user's numbers before reranking.

Done on 3 Nov 2026: citation parsing hardened, retrieval trace explains every inclusion and exclusion, refusals labelled internally, permission filter re-applied locally, offline regression tests added.
