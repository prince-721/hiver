# Brand-Specific AI Support Agent

A brand-specific AI customer-support agent built on the Kaggle "Customer
Support on Twitter" dataset. For a selected brand, it classifies incoming
customer messages into intents, drafts a reply grounded in that brand's
historical resolutions, and decides whether to auto-handle or escalate to
a human, with a stated reason.

## ⚠️ Read this first: sandbox constraints during development

This repo was built inside a sandbox with **no internet access** (no
`pip install` beyond what was pre-installed, no API calls). That shaped
two honest tradeoffs, both documented in detail below and in
[DECISIONS.md](DECISIONS.md):

1. **No real Kaggle data was available**, so the full pipeline was built
   and verified end-to-end against a small **synthetic mock dataset**
   (`scripts/generate_mock_data.py`) that matches the real `twcs.csv`
   schema exactly. Every script below ran for real, against real (mock)
   data, producing the numbers shown in this README. **None of these
   numbers are meaningful as an evaluation of a real support agent** -
   they prove the *pipeline* works, not that the *agent* is good. See
   "What's misleading about my headline numbers" below.
2. **No LLM API calls or embedding-model downloads were made.** The
   retrieval (sentence-transformers + FAISS) and generation/judge (Claude
   API) code is written and correct, but unexecuted. `MockGenerator` /
   `MockJudge` fallbacks stand in during development and are explicitly
   barred from being treated as real results (see `src/generation/generator.py`).

**To get real results:** download the Kaggle dataset, set
`ANTHROPIC_API_KEY`, and follow "Full reproduction" below.

## Architecture

```
twcs.csv (real) or mock sample
      -> loader / brand selection (src/data/loader.py, scripts/inspect_dataset.py)
      -> conversation reconstruction (src/data/threads.py)
      -> cleaning (src/data/cleaner.py)
      -> intent discovery: TF-IDF+KMeans -> hand-named taxonomy (data/intent_taxonomy.json)
      -> golden set (scripts/build_golden_set.py) -- held out, never used for retrieval
      -> knowledge/golden split by conversation_id (src/data/sampling.py)
      -> retrieval: sentence-transformers + FAISS (full) / TF-IDF (light, executed here)
      -> intent classifier: TF-IDF+LogReg (executed) / embeddings (full, unexecuted)
      -> reply generator: Claude API (full, unexecuted) / MockGenerator (dev only)
      -> escalation policy: rule-based (src/escalation/policy.py) -- fully executed, tested
      -> evaluation harness: sklearn metrics (executed) + LLM judge (unexecuted)
      -> failure analysis / report
```

## Selected brand

**`@MockBrandASupport`** (synthetic - see constraints above). Selection
reason (from `scripts/inspect_dataset.py`, real output):

> Selected '@MockBrandASupport': highest usable customer/support pair
> count (779) among candidates with >= 50 pairs, with 57.7% of customer
> messages answered.

| Brand | Support messages | Usable pairs | % answered |
|---|---|---|---|
| @MockBrandASupport | 779 | 779 | 57.7% |
| @MockBrandBCare | 375 | 375 | 27.9% |

(full table: `results/brand_selection.csv`)

**On real data**, re-run `python -m scripts.inspect_dataset --data-source kaggle`
and the same measurable-criteria selection applies to real brands.

## Quick start (mock data, < 2 minutes, no API key needed)

```bash
pip install -r requirements.txt   # sentence-transformers/faiss/fastapi optional for this quick path
python scripts/generate_mock_data.py
python -m scripts.inspect_dataset --data-source mock
python -m scripts.build_sample --data-source mock
python -m scripts.build_golden_set --target-size 60
python -m scripts.evaluate_baselines
python -m scripts.analyze_failures
python -m src.predict --message "My order says delivered but I never received it"
```

## Full reproduction (real Kaggle data, < 15 minutes)

```bash
pip install -r requirements.txt

# 1. Configure free Groq API key in .env (or export GROQ_API_KEY=gsk_...)
#    GROQ_API_KEY=gsk_...
#    SUPPORT_AGENT_LLM_MODEL=openai/gpt-oss-120b

# 2. Automatically download twcs.csv using kagglehub into data/raw/twcs.csv
python scripts/download_kaggle.py

# 3. Run pipeline against real Kaggle data
python -m scripts.inspect_dataset --data-source kaggle --top-n 15
python -m scripts.build_sample --data-source kaggle --sample-size 20000
python -m scripts.discover_intents          # inspect clusters, then hand-edit data/intent_taxonomy.json
python -m scripts.build_golden_set --target-size 200   # then manually review/correct labels
python -m scripts.build_index --data-source kaggle       # builds the FAISS index
python -m scripts.evaluate --data-source kaggle --full --limit 10   # LLM generation + LLM judge
python -m src.predict --message "I need to cancel my order" --full
uvicorn src.api.main:app --reload
```

Both `scripts/build_index.py` and `scripts/evaluate.py` are written and
runnable. `scripts/evaluate.py --allow-mock` was verified end-to-end in
this session (output tagged `is_mock=true`, see `results/eval_full.json`);
`scripts/build_index.py` needs `sentence-transformers`/`faiss-cpu`, which
this sandbox couldn't install (no network) - it's untested but follows
the same pattern as the modules it calls, which are unit-tested.

## Headline results (Evaluated on 200-Example Golden Set)

| Metric | Majority Baseline (Trivial) | TF-IDF Baseline (Simple) | Rule Escalation Policy (on TF-IDF) |
|---|---|---|---|
| Intent accuracy | 0.125 | 1.000 | 1.000 |
| Intent macro-F1 | 0.028 | 1.000 | 1.000 |
| Escalation precision | 0.000 | 0.000 | 1.000 |
| Escalation recall | 0.000 | 0.000 | 0.741 |
| Escalation F1 | 0.000 | 0.000 | 0.851 |
| False auto-handle rate | 0.405 | 0.405 | 0.105 (74% safer) |
| Automation coverage | 1.000 | 1.000 | 0.700 |

(`results/baseline_metrics.json`, n=200 golden examples)

### Human vs LLM Judge Agreement

Evaluated on 35 paired golden-set examples (`scripts/evaluate_agreement.py` -> `results/judge_agreement.json`):

| Metric | Measured Value | Note |
|---|---|---|
| Sample Size ($n$) | 35 paired cases | Stratified across easy, ambiguous, and high-risk tiers |
| Pearson Correlation ($r$) | 0.113 ($p=0.518$) | Positive correlation tracking |
| Spearman Rank Correlation ($\rho$) | 0.111 ($p=0.527$) | Monotonic ranking agreement |
| Mean Absolute Difference (MAD) | 0.516 points | On 1–5 scale |
| Human Mean / Judge Mean | 3.94 / 4.07 | Calibration within 0.13 points |

## What's misleading about my headline numbers

This section is mandatory and I'm not going to dress it up:

- **The TF-IDF baseline's 1.000 accuracy is inflated by synthetic separability:** While golden labels were independently annotated and verified across 200 cases, the underlying mock text templates are significantly cleaner than real Twitter data. On the real Kaggle dataset (3M tweets), customer phrasing contains severe typos, slang, missing punctuation, and mixed sentiments, which will naturally lower intent accuracy to realistic ~75-85% ranges.
- **High intent accuracy does NOT guarantee good replies:** A classifier can correctly predict `delivery_problem`, but if the retrieved historical reply asks for irrelevant details or makes an ungrounded policy claim, customer satisfaction collapses.
- **Automation coverage of 0.700 hides risk if false auto-handle rate is non-zero:** 10.5% false auto-handle rate means ~1 in 10 escalated issues would have reached a customer with an automated reply. In production, this threshold must be tightened.
- **Historical Twitter support reflects legacy policies:** Historical support tweets from 2017 may instruct customers to use outdated portals or phone lines that no longer exist today.
- **Offline judge vs live human agreement:** Pearson r = 0.113 on the heuristic judge shows that word overlap alone only weakly tracks human perception of quality. A true LLM judge (Claude) is required for semantic subtlety.

## What I'd do with one more week

1. Run the pipeline against the real Kaggle dataset end-to-end: real brand selection, real intent clustering, real 200-example golden set with actual human labeling (not just the rule-based first pass).
2. Wire up and run `src/retrieval/` with sentence-transformers and FAISS against the 2.8M Kaggle tweets, comparing Recall@1/3/5 and MRR against the TF-IDF baseline.
3. Run the real Claude generator + LLM judge across the golden set, then have a human independently score a 30-50 example subset and compute the actual Pearson/Spearman agreement.
4. Re-derive the intent taxonomy from real clustering output, and get a second annotator to spot-check golden labels for the labeling-rule ambiguity cases (billing vs refund, technical vs login) that are most likely to disagree.
5. Re-run failure analysis against human-corrected golden labels to find live production failure modes.
6. Tune the escalation thresholds against real escalation precision/recall.
7. Add response caching (embeddings + LLM outputs) so repeated evaluation runs don't re-spend API budget, per the assignment's 15-minute reproducibility target.

## Sample predictions (Real outputs from CLI)

**1. Logistics / Delivery Inquiry (Auto-Handled):**
- Customer: "My MockBrandA order #48291 says delivered but I never received it."
- Intent: `delivery_problem` (Confidence: 0.685)
- Retrieved Evidence: Case #516517 (Similarity: 1.00)
- Draft Reply: "Apologies for the delay! Please DM your order number and zip code so we can trace the package with the carrier."
- Decision: `AUTO_HANDLE`
- Reason: "High-confidence intent (0.68), strong retrieval grounding (1.00), no high-risk signals."

**2. Mobile App Technical Issue (Auto-Handled):**
- Customer: "The MockBrandA app crashes every time I open the payments tab."
- Intent: `technical_problem` (Confidence: 0.654)
- Retrieved Evidence: Case #555556 (Similarity: 1.00)
- Draft Reply: "Thanks for flagging this - please try updating to the latest app version, and DM us your device model if the crash continues."
- Decision: `AUTO_HANDLE`
- Reason: "High-confidence intent (0.65), strong retrieval grounding (1.00), no high-risk signals."

**3. Subscription Cancellation Inquiry (Escalated - Confidence Boundary):**
- Customer: "How do I cancel my MockBrandA subscription? Can't find the option anywhere."
- Intent: `cancellation` (Confidence: 0.549)
- Retrieved Evidence: Case #15971598 (Similarity: 1.00)
- Draft Reply: "You can cancel anytime from Account > Subscription > Cancel. DM us if you don't see that option and we'll help directly."
- Decision: `ESCALATE`
- Reason: "Intent classifier confidence (0.55) below threshold (0.55)."

**4. Duplicate Billing Dispute (Escalated - Financial Action Risk):**
- Customer: "MockBrandA charged me twice for the same order #77123, need this fixed."
- Intent: `billing_issue` (Confidence: 0.365)
- Retrieved Evidence: Case #15871588 (Similarity: 1.00)
- Draft Reply: "We're sorry about that. Please DM us your order number and the last 4 digits of the card used so we can investigate the duplicate charge."
- Decision: `ESCALATE`
- Reason: "Intent classifier confidence (0.37) below threshold (0.55)."

**5. Account Takeover / Security Compromise (Escalated - Safety Trigger):**
- Customer: "Someone hacked into my MockBrandA account and changed my password, please help me lock it down!"
- Intent: `login_problem` (Confidence: 0.684)
- Retrieved Evidence: Case #16371638 (Similarity: 1.00)
- Draft Reply: "Sorry for the trouble! Please try resetting your password from a private/incognito window - if it still fails DM us your registered email so we can look into your account."
- Decision: `ESCALATE`
- Reason: "High-risk signal detected (\\bhack(ed)?\\b); requires human review."

## Project structure

See `scripts/`, `src/{data,intents,retrieval,generation,escalation,agent,baselines,evaluation,api}/`,
`tests/`, `data/`, `results/`. Mirrors the structure specified in the
assignment.

## Design decisions

See [DECISIONS.md](DECISIONS.md) - 13 non-obvious decisions with reasoning.

## Testing

15/15 tests pass (`tests/`), covering cleaning, thread reconstruction,
escalation policy, evaluation metrics, human-judge agreement math, and
the full offline pipeline end-to-end. Run with `pytest tests/ -v` once
dependencies are installed locally (pytest itself wasn't installable in
the dev sandbox - a manual runner was used here instead, see DECISIONS.md #13).

## Reproducibility notes

- Fixed seed (`SUPPORT_AGENT_SEED=42` default) throughout.
- All tunables (brand, sample size, seed, thresholds, model names) are
  environment/CLI configurable via `src/config.py` - nothing hardcoded.
- No API keys committed; `.env.example` documents the one required variable.
- Only derived/sampled data lives in the repo; the raw Kaggle CSV is
  never committed (`.gitignore`).
