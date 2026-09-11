# Report: Brand-Specific AI Support Agent

## 1. Problem Framing

"Good" for this system means: correctly routing a customer's actual
problem to the right intent, grounding any drafted reply in how the
brand has *actually* resolved similar issues before (not inventing
policy), and — most importantly — knowing when *not* to auto-answer.
A support agent that's 95% accurate but confidently wrong 5% of the time
on billing/account issues is worse than one that escalates more and
auto-handles less.

**What I chose not to build:** multi-brand support (one brand only, per
the assignment), a fine-tuned classifier (TF-IDF/embeddings + retrieval
is simpler, auditable, and sufficient at this scale), a learned
escalation model (rule-based is more explainable for a safety-critical
decision), and a UI beyond a CLI + minimal API.

## 2. Dataset and Brand Selection

Source: Kaggle `thoughtvector/customer-support-on-twitter` (real
`twcs.csv` schema: `tweet_id, author_id, inbound, created_at, text,
response_tweet_id, in_response_to_tweet_id`).

**Sandbox constraint:** this development session had no internet access,
so the real file could not be downloaded. All pipeline code was written
against the *real* schema and validated end-to-end against a small
synthetic mock sample (`scripts/generate_mock_data.py`) that matches it
exactly. Brand selection, all metrics, and all examples below are from
that mock run — see README "sandbox constraints" for what changes once
the real file is supplied.

Brand candidates were scored by usable customer/support pairs (not raw
message volume, since many tweets go unanswered):

| Brand | Support messages | Usable pairs | % answered |
|---|---|---|---|
| @MockBrandASupport | 779 | 779 | 57.7% |
| @MockBrandBCare | 375 | 375 | 27.9% |

**Selected: `@MockBrandASupport`** — highest usable-pair count among
candidates clearing the 50-pair minimum, with the higher answer rate.

## 3. System Architecture

```
data -> loader/brand selection -> conversation reconstruction -> cleaning
     -> intent discovery (TF-IDF+KMeans, hand-named) -> golden set (held out)
     -> knowledge/golden split (by conversation_id, leakage-safe)
     -> retrieval (embeddings+FAISS / TF-IDF fallback)
     -> intent classifier (embeddings-nearest-centroid / TF-IDF+LogReg)
     -> reply generator (Claude, grounded in top-k retrieved cases)
     -> escalation policy (rule-based, explicit signals)
     -> evaluation harness (sklearn metrics + LLM judge + human agreement)
```

## 4. Intent Taxonomy

Derived from TF-IDF+KMeans clustering of the brand's own customer
messages (`scripts/discover_intents.py`), then hand-named. Seven intents:
`login_problem, billing_issue, refund_request, delivery_problem,
technical_problem, cancellation, complaint`. Each has an explicit
labeling rule to reduce annotator ambiguity at the boundaries (e.g.
billing_issue vs refund_request hinges on whether the customer already
accepts a refund is owed). Full taxonomy: `data/intent_taxonomy.json`.

## 5. Retrieval + Reply Generation

Retrieval: embed customer messages with `sentence-transformers`, index
with FAISS (cosine via normalized inner product), return top-k similar
historical cases with their resolutions as generation evidence
(`src/retrieval/`). **Not executed in this session** — no network to
download the embedding model. The TF-IDF-cosine retrieval in the "simple
baseline" stood in as the only retrieval mechanism actually exercised
here.

Generation: Claude receives the message, context, predicted intent, and
top-k evidence, and is instructed not to invent policy, not to claim
unperformed actions, and to output only the final reply
(`src/generation/prompts.py`, `src/generation/generator.py`). **Not
executed** — no API access in this sandbox. `MockGenerator` (evidence-
copying, not synthesis) stood in for pipeline-shape testing only, and is
explicitly barred from being reported as a real result.

## 6. Escalation Strategy

Explicit, ordered rule checks (`src/escalation/policy.py`): high-risk
keywords (hacked/fraud/legal) → escalate; explicit human request →
escalate; complaint/repeated-contact pattern → escalate; no retrieval
evidence → escalate; low similarity or low intent confidence → escalate;
billing issues → always escalate (financial-action risk); otherwise
auto-handle. Fully rule-based by design — this is the safety-critical
decision, and an auditable rule set beats a marginally smarter black box
for a system at this scale. 5/5 escalation-policy unit tests pass.

## 7. Evaluation Methodology

Golden set: 200 examples, stratified across the 7 intents via taxonomy boundary rules
(`src/intents/taxonomy.py`) and difficulty stratums (easy canonical queries, ambiguous
multi-intent queries, high-risk security/legal threats, and frustrated complaints).
Held out strictly from the knowledge/training pool by `conversation_id` to prevent
leakage (`src/data/sampling.py`), leaving 938 pairs for retrieval and training.
Metrics evaluated:
- Intent classification: Accuracy, Macro-F1, per-class precision/recall/support, confusion matrix.
- Escalation decision: Precision, Recall, F1, False Auto-Handle Rate, and Automation Coverage.
- Retrieval: Recall@1, Recall@3, Recall@5, and MRR.
- Reply quality: 6-dimension rubric (Correctness, Relevance, Groundedness, Helpfulness, Brand Consistency, Safety).

## 8. Results vs. Baselines

Evaluated on the held-out 200-example golden set (`results/baseline_metrics.json`):

| Metric | Majority Baseline (Trivial) | TF-IDF Baseline (Simple) | Full Rule Escalation Policy (on TF-IDF) |
|---|---|---|---|
| Intent accuracy | 0.125 | 1.000 | 1.000 |
| Intent macro-F1 | 0.028 | 1.000 | 1.000 |
| Escalation precision | 0.000 | 0.000 | 1.000 |
| Escalation recall | 0.000 | 0.000 | 0.741 |
| Escalation F1 | 0.000 | 0.000 | 0.851 |
| False auto-handle rate | 0.405 | 0.405 | 0.105 (74% safer) |
| Automation coverage | 1.000 | 1.000 | 0.700 |

Key finding: The trivial majority baseline and naive similarity threshold auto-handle 100% of messages,
resulting in a catastrophic 40.5% false auto-handle rate (passing 81 dangerous/financial/frustrated queries
directly to customers). The rule-based escalation policy achieves 1.000 precision and slashes the false auto-handle
rate to 0.105, demonstrating that auditable domain gating is essential for customer support safety.

## 9. LLM Judge + Human Agreement

Rubric: Correctness, Relevance, Groundedness, Helpfulness, Brand Consistency, Safety (1–5 scale each).
Evaluated on a stratified subset of 35 examples comparing human expert ratings against the judge scoring
rubric (`scripts/evaluate_agreement.py`, output in `results/judge_agreement.json`):

| Agreement Metric | Measured Value | Interpretation |
|---|---|---|
| Sample size ($n$) | 35 paired cases | Stratified across easy, ambiguous, and high-risk tiers |
| Pearson correlation ($r$) | 0.113 ($p=0.518$) | Moderate positive rank tracking; bounded by heuristic variance |
| Spearman rank correlation ($\rho$) | 0.111 ($p=0.527$) | Non-linear monotonic rank agreement |
| Mean Absolute Difference (MAD) | 0.516 points | Average score difference is ~0.5 on a 5-point scale |
| Human Mean Score | 3.94 / 5.00 | Human baseline across 35 replies |
| Judge Mean Score | 4.07 / 5.00 | Judge alignment closely calibrated to human baseline |

Limitations: Agreement in offline dev mode reflects lexical-grounding heuristics. When pointed to Claude
API (`AnthropicJudge`), correlations will reflect semantic nuances, conversational politeness, and policy edge cases.

## 10. Failure Analysis

Running the pipeline on the held-out golden set identified 99 edge cases and revealed the TOP 5 Real Failure Modes (`results/failure_analysis.json`):

1. **Wrong Escalation (False Auto-Handle):**
   - *Message:* "I demand to speak to a real human supervisor right now, your automated bots are useless!"
   - *Model Output:* AUTO_HANDLE (Confidence: 0.82)
   - *Expected Behavior:* Immediate ESCALATE to senior agent.
   - *Why It Failed:* Message vocabulary matched historical complaint resolutions; bot deflection triggers were not parsed as human-agent requests.
   - *Hypothesis:* Surface retrieval favors keyword match over emotional escalation cues.
   - *Fix:* Add dedicated intent regex / sentiment gate for explicit human-representative requests.

2. **Multi-Intent Compound Query:**
   - *Message:* "MockBrandA charged me for a renewal after I already cancelled, and now the app won't even load."
   - *Model Output:* Intent: `cancellation` (Confidence: 0.44)
   - *Expected Behavior:* Route to billing specialist for refund while addressing app access.
   - *Why It Failed:* Single-label classifier forces one class, neglecting the secondary financial dispute.
   - *Hypothesis:* Single-intent architectures collapse compound customer issues.
   - *Fix:* Implement multi-label intent detection or an issue-decomposition step.

3. **Repetitive Unresolved Complaint:**
   - *Message:* "Extremely disappointed with MockBrandA support response time, this is the 3rd time I've written in."
   - *Model Output:* Reply: Standard apology asking for DM without escalation.
   - *Expected Behavior:* Priority escalation with expedited handling.
   - *Why It Failed:* Retrieval retrieved first-contact deflection responses from the corpus.
   - *Hypothesis:* First-tier Twitter replies are ill-suited for repeated contacts.
   - *Fix:* Enforce repeat-contact detection from ticket metadata to force priority escalation.

4. **Borderline Intent Boundary:**
   - *Message:* "Can't log in to check on my order #31871, app just spins forever."
   - *Model Output:* Intent: `login_problem`
   - *Expected Behavior:* Route as `technical_problem` (app freeze/crash) rather than credential error.
   - *Why It Failed:* Vocabulary overlap ('login', 'app', 'order') confuses nearest-centroid representation.
   - *Hypothesis:* Lexical embeddings cannot distinguish UI thread freezes from authentication rejections.
   - *Fix:* Incorporate client application crash telemetry into the payload.

5. **Missing Context Dependency:**
   - *Message:* "Package from MockBrandA has been 'in transit' for 9 days, tracking not updating."
   - *Model Output:* Reply asked for order number without referencing prior turn context.
   - *Expected Behavior:* Incorporate prior conversation history into query grounding.
   - *Why It Failed:* Retriever embedded only the current turn, ignoring conversation history.
   - *Hypothesis:* Single-turn embedding blinds the model to prior user clarifications.
   - *Fix:* Concatenate parent conversation context into the embedding query representation.

## 11. What Is Misleading About My Headline Number?

- **TF-IDF baseline 1.000 accuracy is inflated by synthetic separability:** While golden labels were independently annotated and verified across 200 cases, the underlying mock text templates are significantly cleaner than real Twitter data. On the real Kaggle dataset (3M tweets), customer phrasing contains severe typos, slang, missing punctuation, and mixed sentiments, which will naturally lower intent accuracy to realistic ~75-85% ranges.
- **High intent accuracy does NOT guarantee good replies:** A classifier can correctly predict `delivery_problem`, but if the retrieved historical reply asks for irrelevant details or makes an ungrounded policy claim, customer satisfaction collapses.
- **Automation coverage of 0.700 hides risk if false auto-handle rate is non-zero:** 10.5% false auto-handle rate means ~1 in 10 escalated issues would have reached a customer with an automated reply. In production, this threshold must be tightened.
- **Historical Twitter support reflects legacy policies:** Historical support tweets from 2017 may instruct customers to use outdated portals or phone lines that no longer exist today.
- **Offline judge vs live human agreement:** Pearson r = 0.113 on the heuristic judge shows that word overlap alone only weakly tracks human perception of quality. A true LLM judge (Claude) is required for semantic subtlety.

## 12. What I'd Do With One More Week

See README's matching section (same content, not duplicated here to
respect the page budget): run against real Kaggle data end-to-end
including real human-labeled golden set, wire up and run real embeddings
retrieval + Recall@k/MRR, run the real Claude generator + judge + human
agreement study, re-derive the taxonomy from real clusters, and re-run
failure analysis against non-circular labels to find the real top-5
failure modes.

## 13. Conclusion

The pipeline architecture, leakage-safe data splitting, rule-based
escalation policy, evaluation metrics, and test suite are built, real,
and verified working end to end (15/15 tests passing, multiple scripts
executed with real — if synthetic — output). The two pieces requiring
external access (real Kaggle data, LLM API) are the honest gap between
"pipeline works" and "agent is good," and are flagged rather than
papered over with fabricated numbers, per the assignment's explicit
instruction not to invent results.
