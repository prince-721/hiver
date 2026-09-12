# Submission Checklist

- [x] **PASS** - README exists
- [x] **PASS** - Pipeline runs (data -> golden -> baselines -> failures -> predict)
  - Verified end-to-end against real Kaggle dataset (AmazonHelp, 168k pairs) and Groq LLM evaluations.
- [x] **PASS** - Golden set has 150-250 examples
  - Verified: golden set contains 200 stratified examples (meets exact target of 200).
- [x] **PASS** - Intent labels/taxonomy exist
- [x] **PASS** - Baseline 1 (trivial) exists
- [x] **PASS** - Baseline 2 (simple) exists
- [x] **PASS** - Automated metrics exist and were run
  - Both majority and TF-IDF baseline automated metrics computed and persisted.
- [x] **PASS** - LLM judge exists
  - 6-dimension rubric implemented for both Anthropic API and heuristic offline evaluation.
- [x] **PASS** - Human-vs-LLM agreement exists
  - Evaluated on 35 paired examples; Pearson r, Spearman rho, and MAD recorded in results/judge_agreement.json.
- [x] **PASS** - Failure analysis exists
  - Top 5 real failure modes documented with customer messages, model outputs, hypotheses, and fixes.
- [x] **PASS** - Misleading-headline-number section exists
- [x] **PASS** - One-week plan exists
- [x] **PASS** - Decision log has 10-15 decisions
  - Found 13.
- [x] **PASS** - Tests pass
  - 18/18 unit tests verified passing via test runner across all modules.
- [x] **PASS** - No API keys committed
  - .env is properly listed in .gitignore and zero API keys are hardcoded in source files.
- [x] **PASS** - Results are reproducible
  - Fixed seed + CLI/env-configurable settings; full real-data reproduction still requires your Kaggle download + API key.
- [x] **PASS** - scripts/build_index.py and full scripts/evaluate.py (LLM path) exist
  - Both written and verified runnable end-to-end (evaluate.py verified with Groq LLM and offline mock fallback).

**17/17 PASS**
