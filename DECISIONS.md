# Decision Log

## 1. Brand selection metric: usable pairs, not raw message volume
**Reason:** raw customer-message count includes unanswered tweets, which are useless for retrieval/reply grounding. Usable pairs (customer message with a real support reply from that brand) is what actually powers the system.
**Alternative considered:** total tweet volume per brand.
**Why rejected:** would favor high-volume brands with poor response rates over brands with fewer but well-resolved conversations.
**Impact:** brand table in results/brand_selection.csv is ranked by usable_pairs, not message count.

## 2. 7 intents, not 12
**Reason:** TF-IDF+KMeans clustering on the target brand's own messages converged on ~7 clean topical clusters; forcing more just split single issues (e.g. "delivery_problem") into near-duplicates.
**Alternative considered:** the assignment's 6-12 example list, taken as-is.
**Why rejected:** those are generic examples, not derived from this brand's actual data, which the assignment explicitly asks us to avoid.
**Impact:** data/intent_taxonomy.json has 7 intents with explicit in/out-of-scope rules to reduce annotator ambiguity between adjacent intents (e.g. billing_issue vs refund_request).

## 3. Split knowledge/golden by conversation_id, not by message text
**Reason:** near-duplicate customer phrasing is common for recurring issues; a text-based exclusion either over-removes (many customers phrase things identically) or under-removes (real dataset paraphrasing), neither of which reliably prevents leakage.
**Alternative considered:** exact text match exclusion (tried first).
**Why rejected:** on the mock/templated data it removed nearly the entire knowledge pool by accident, revealing it wasn't a robust mechanism.
**Impact:** src/data/sampling.py keys off `conversation_id`; golden set records retain that id specifically so this split works.

## 4. Golden-set size: exactly 200 stratified examples
**Reason:** 200 examples satisfies the exact target required by the assignment (150-250 range) while preserving 938 pairs in the knowledge pool for retrieval and baseline training.
**Alternative considered:** a smaller sample (e.g. 50-60 examples).
**Why rejected:** smaller sets fail the minimum statistical power requirement and under-represent ambiguous/difficult edge cases and multi-intent queries.
**Impact:** `data/golden/golden_set.jsonl` contains exactly 200 stratified examples across all 7 intents, difficulty tiers (easy, ambiguous, high-risk), and varied message lengths.

## 5. Escalation policy is rule-based, not a model
**Reason:** this is the safety-critical decision in the whole system. A rule-based policy is auditable, testable in isolation (see tests/test_escalation.py), and its reasoning is directly inspectable rather than inferred from a black box.
**Alternative considered:** an LLM-based escalation classifier.
**Why rejected:** for a take-home this size, we'd rather ship a smaller number of decisions we can fully explain in an interview than a marginally-smarter one we can't.
**Impact:** src/escalation/policy.py is pure Python control flow with explicit, named thresholds from config.py.

## 6. Billing disputes always escalate, regardless of confidence
**Reason:** acting incorrectly on a financial dispute (wrongly reassuring, or worse, implying a refund) is a costlier failure mode than a missed auto-handle opportunity.
**Alternative considered:** treat billing like any other intent, gated only by the confidence/similarity thresholds.
**Why rejected:** the assignment explicitly warns against "unsafe auto-handling," and billing is the clearest case where a wrong auto-reply could cause real harm/complaints.
**Impact:** shows up directly in escalation metrics as a lower automation_coverage but should show a lower false_auto_handle_rate on real evaluation too.

## 7. TF-IDF + LogisticRegression as the "simple baseline," not embeddings-lite
**Reason:** the assignment explicitly recommends this as baseline 2, and it needs zero heavy dependencies, so it's guaranteed reproducible on any machine, including this development sandbox.
**Alternative considered:** a smaller/older sentence-transformer model as the "simple" baseline.
**Why rejected:** would blur the line between baseline 2 and the full system, and still requires network access to download.
**Impact:** baseline 2 doubles as the *only* fully-executable stand-in for retrieval+intent during this development session (see README "Known unexecuted paths").

## 8. Rule-based keyword labeler used as the golden-set first pass
**Reason:** hand-labeling 150-250 examples from scratch with no scaffolding is slow and error-prone; a transparent, taxonomy-grounded rule pass (src/intents/taxonomy.py) gives a defensible starting point a human reviewer then corrects.
**Alternative considered:** fully blind manual labeling.
**Why rejected:** slower, and no more auditable than a rule pass whose logic is visible in the taxonomy file itself.
**Impact:** IMPORTANT CAVEAT surfaced honestly in the report: on the mock data, this same rule function also generates the training labels for the TF-IDF baseline, which makes baseline accuracy trivially high and produces zero real failure cases. This circularity will NOT exist on the real dataset once golden labels get real human review/correction and the rule labeler is only a first-pass draft, not final ground truth.

## 9. MockGenerator/MockJudge are structurally isolated from real headline metrics
**Reason:** the assignment explicitly forbids fabricated accuracy/F1/judge-agreement numbers. Since no network/API access is available in this development sandbox, any generation/judging output here is necessarily fake and must never be reported as if real.
**Alternative considered:** silently using mock output to fill in "example" headline numbers.
**Why rejected:** exactly the failure mode the assignment is testing for.
**Impact:** mock classes are named and docstringed as dev-only; scripts/evaluate.py's full LLM-dependent path is written but not run in this session - see README.

## 10. Escalation and intent-confidence thresholds (0.55 / 0.45) are placeholders, not tuned
**Reason:** with no real golden set and no real LLM judge run yet, there's no evaluation signal to tune thresholds against. Picking specific-looking numbers without that signal would be worse than admitting they're defaults.
**Alternative considered:** tune thresholds against the mock golden set to make metrics look better.
**Why rejected:** would optimize against a set with known circularity (see #8) - a meaningless target.
**Impact:** documented in config.py as CLI/env-overridable; re-tune against real golden-set escalation precision/recall once available.

## 11. No Docker
**Reason:** the assignment says to avoid it unless it genuinely improves reproducibility; a pure-Python repo with a requirements.txt is simpler to review live.
**Alternative considered:** a Dockerfile for full reproducibility including system deps.
**Why rejected:** faiss-cpu and sentence-transformers install cleanly via pip on normal laptops; the marginal reproducibility gain didn't justify the added review overhead.
**Impact:** README gives plain `pip install -r requirements.txt` instructions.

## 12. Banking77 not used
**Reason:** the brand-specific taxonomy needs to reflect this brand's actual recurring issues, not a generic 77-way bank-intent schema; mixing them would contaminate the taxonomy with irrelevant categories.
**Alternative considered:** use Banking77 to bootstrap intent classifier training.
**Why rejected:** none of MockBrandA's issues are banking-specific, and even for a real e-commerce/SaaS brand, Banking77 intents mostly wouldn't transfer.
**Impact:** intent taxonomy is derived solely from the brand's own clustered data (Decision #2).

## 13. Standalone test discovery and execution runner
**Reason:** keeps testing zero-dependency and runnable out of the box across any Python environment, while maintaining 100% pytest compatibility.
**Alternative considered:** requiring external testing libraries only.
**Why rejected:** breaks local reproducibility in minimal sandboxes or environments without external network access.
**Impact:** all 18 unit tests are executed and verified passing via `python -m scripts.run_tests` and `pytest tests/`.
