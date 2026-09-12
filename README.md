# Brand-Specific AI Customer Support Agent

An end-to-end, brand-specific AI customer support agent built on the Kaggle **Customer Support on Twitter** dataset (2.8M multi-turn tweets). For the target brand (**`AmazonHelp`**), the system:
1. **Classifies** incoming customer inquiries into a data-grounded intent taxonomy.
2. **Drafts grounded replies** conditioned on historical resolutions retrieved from the brand's knowledge base.
3. **Decides** whether to auto-handle or escalate to a human supervisor with an explicit, auditable reason.
4. **Evaluates** response quality across a 6-dimension LLM judge rubric with human-agreement calibration.

---

## 🏛️ System Architecture

```
twcs.csv (Kaggle, 2.8M tweets)
    │
    ├──> 1. Dataset Ingestion & Brand Scoring (scripts/inspect_dataset.py)
    │        Selected brand: AmazonHelp (168,814 usable customer/support pairs)
    │
    ├──> 2. Thread Reconstruction & Cleaning (src/data/threads.py, cleaner.py)
    │        Vectorized parent-child conversation matching (19,553 clean pairs)
    │
    ├──> 3. Intent Discovery (scripts/discover_intents.py)
    │        TF-IDF + KMeans clustering -> 7-intent taxonomy (data/intent_taxonomy.json)
    │
    ├──> 4. Golden Evaluation Set (data/golden/golden_set.jsonl)
    │        200 stratified hand-labeled cases held out strictly by conversation_id
    │
    ├──> 5. Retrieval Engine (src/retrieval/)
    │        TF-IDF cosine similarity + Vector index retriever over historical resolutions
    │
    ├──> 6. Intent Classifier & Escalation Policy (src/intents/, src/escalation/)
    │        Multi-class classification + rule-based safety gating (fraud/legal/financial checks)
    │
    ├──> 7. Grounded Reply Generator (src/generation/)
    │        Groq API (openai/gpt-oss-120b) conditioned on retrieved historical cases
    │
    └──> 8. Evaluation Harness & LLM Judge (src/evaluation/)
             6-dimension rubric + Pearson/Spearman human agreement calibration
```

---

## 🎯 Selected Brand: `AmazonHelp`

Selected automatically using measurable criteria from [scripts/inspect_dataset.py](scripts/inspect_dataset.py) over the entire Kaggle corpus:

| Rank | Brand | Support Messages | Usable Pairs | % Customer Msgs Answered | Avg Msg Length |
|:---:|---|:---:|:---:|:---:|:---:|
| **1** | **AmazonHelp** | **169,840** | **168,814** | **10.98%** | **116.6 chars** |
| 2 | AppleSupport | 106,860 | 106,646 | 6.93% | 109.3 chars |
| 3 | Uber_Support | 56,270 | 56,160 | 3.65% | 120.2 chars |
| 4 | SpotifyCares | 43,265 | 43,092 | 2.80% | 103.9 chars |
| 5 | Delta | 42,253 | 42,114 | 2.74% | 110.7 chars |

*(Full scored ranking saved in `results/brand_selection.csv`)*

---

## 🚀 Quick Start (< 2 Minutes)

All unit tests and offline evaluation run in seconds with zero setup:

```powershell
# 1. Install dependencies
py -3.13 -m pip install -r requirements.txt

# 2. Run automated test suite (18/18 passing)
py -3.13 -m pytest tests/ -v

# 3. Verify submission criteria (17/17 PASS)
py -3.13 scripts/check_submission.py

# 4. Run fast offline evaluation (~1.5s across 200 golden cases)
py -3.13 -m scripts.evaluate --allow-mock
```

---

## 📊 Full Reproduction with Real Kaggle Data & Groq LLM

```powershell
# 1. Download Kaggle dataset automatically via kagglehub into data/raw/twcs.csv
py -3.13 scripts/download_kaggle.py

# 2. Inspect brands and score candidates
py -3.13 -m scripts.inspect_dataset --data-source kaggle --top-n 15

# 3. Extract and clean 20,000 conversation pairs for AmazonHelp
py -3.13 -m scripts.build_sample --data-source kaggle --sample-size 20000

# 4. Discover topical clusters from real customer tweets
py -3.13 -m scripts.discover_intents --n-clusters 7

# 5. Run live end-to-end prediction with Groq LLM
py -3.13 -m src.predict --message "My Amazon package says delivered but I never received it" --full

# 6. Run live evaluation with Groq LLM generation and 6-dimension LLM judge
py -3.13 -m scripts.evaluate --data-source kaggle --full --limit 10
```

---

## 📈 Headline Results

### 1. Intent Classification & Escalation Safety (Golden Set)

| Metric | Majority Baseline (Trivial) | TF-IDF Baseline (Simple) | Rule Escalation Policy (on Real Data) |
|---|:---:|:---:|:---:|
| **Intent Accuracy** | 0.125 | 1.000 | **0.900** |
| **Intent Macro-F1** | 0.028 | 1.000 | **0.810** |
| **Escalation Precision** | 0.000 | 0.000 | **0.600 – 1.000** |
| **Escalation Recall** | 0.000 | 0.000 | **1.000** |
| **False Auto-Handle Rate** | 0.405 | 0.405 | **0.000 (0% dangerous leaks)** |
| **Automation Coverage** | 1.000 | 1.000 | **0.500 – 0.700** |

### 2. Human vs. LLM Judge Agreement

Evaluated on 35 paired cases ([results/judge_agreement.json](results/judge_agreement.json)) across a 6-dimension quality rubric (Correctness, Relevance, Groundedness, Helpfulness, Brand Consistency, Safety):

| Metric | Measured Value | Meaning |
|---|:---:|---|
| **Sample Size ($n$)** | 35 paired cases | Stratified across easy, ambiguous, and high-risk queries |
| **Pearson Correlation ($r$)** | **0.371 ($p = 0.0281$)** | Statistically significant positive correlation with human ratings |
| **Spearman Rank ($\rho$)** | **0.168 ($p = 0.333$)** | Monotonic ranking agreement |
| **Mean Absolute Diff (MAD)** | **1.014 points** | Average delta on 1–5 scoring scale |
| **Human Mean / Judge Mean** | **3.94 / 4.96** | Real human annotations vs. Groq LLM judge |

---

## 🔬 What's Misleading About My Headline Numbers

1. **Synthetic vs. Real Intent Separability**: Synthetic templates yield 1.000 intent accuracy because keywords are distinct. On real Twitter data (where typos, slang, and mixed sentiment abound), accuracy is ~85–90%.
2. **Intent Accuracy $\neq$ Reply Quality**: Correctly classifying `delivery_problem` does not guarantee the reply resolves the customer's problem. Grounding against relevant historical evidence is what determines resolution quality.
3. **The False Auto-Handle Danger**: A 70% automation rate looks impressive on executive dashboards, but if the false auto-handle rate is non-zero, dangerous billing, legal, or compromised-account queries reach customers with canned responses. The escalation policy intentionally prioritizes a **0.00% false auto-handle rate** over higher coverage.
4. **Historical Twitter Drift**: Real tweets from 2017 often reference discontinued tracking links (`^BV`) or outdated self-service portals. Grounded generators must be taught to filter expired policy details.

---

## 🔍 Sample Real Predictions (CLI Outputs)

### 1. Logistics / Delivery Inquiry (Auto-Handled)
- **Customer Query**: *"My Amazon package says delivered but I never received it"*
- **Intent**: `delivery_problem` (Confidence: 0.998)
- **Retrieved Evidence**: Case #331646331645 (Similarity: 0.625)
- **Draft Reply**: *"I’m sorry you haven’t received your package even though it shows as delivered. Please review the tips for locating missing deliveries here: [link]. If it’s still missing, let us know and we’ll investigate further."*
- **Decision**: `AUTO_HANDLE`
- **Reason**: *"High-confidence intent (1.00), strong retrieval grounding (0.63), no high-risk signals."*

### 2. Subscription / Cancellation (Escalated Safety Gating)
- **Customer Query**: *"I need to cancel my order and dispute a charge"*
- **Intent**: `cancellation` (Confidence: 0.675)
- **Decision**: `ESCALATE`
- **Reason**: *"Financial keyword dispute detected; requires human agent review."*

---

## 🗺️ What I'd Do With One More Week

1. **Multi-Label Intent Decomposition**: Customer messages frequently bundle multiple issues (e.g., *"charge me twice and package was broken"*). A multi-label or query-decomposition step would route each sub-issue independently.
2. **Dense Vector Embeddings at Scale**: Build a persistent FAISS index over all 168k `AmazonHelp` pairs using `all-MiniLM-L6-v2` with GPU acceleration.
3. **Response Caching**: Add local SQLite/Redis caching for LLM responses and embeddings to eliminate duplicate API calls and achieve instant 100% reproducible evaluations.
4. **Context Window Concatenation**: Concatenate multi-turn customer thread context into retrieval embeddings to resolve ambiguous follow-up tweets (*"did that, still broken"*).

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/                 # Real Kaggle dataset (twcs.csv) - gitignored
│   ├── golden/              # 200 stratified hand-labeled golden evaluation set
│   ├── processed/           # 19,553 clean AmazonHelp conversation pairs
│   └── intent_taxonomy.json # 7-intent taxonomy with boundary rules
├── results/
│   ├── brand_selection.csv  # Scored ranking of all Kaggle brands
│   ├── eval_full.json       # Live Groq LLM evaluation metrics
│   ├── judge_agreement.json # 35 paired human vs. LLM judge scores
│   └── failure_analysis.json# Top 5 real failure modes with root cause analysis
├── scripts/
│   ├── download_kaggle.py   # Automated kagglehub downloader
│   ├── inspect_dataset.py   # Dataset profiling & brand selection
│   ├── build_sample.py      # Vectorized thread reconstruction & cleaning
│   ├── discover_intents.py  # TF-IDF + KMeans cluster discovery
│   ├── evaluate.py          # Full evaluation harness (metrics + LLM judge)
│   ├── evaluate_agreement.py# Human vs. LLM judge agreement calculation
│   ├── check_submission.py  # Automated 17-point audit checklist
│   └── run_tests.py         # Test runner
├── src/
│   ├── agent/               # End-to-end SupportAgentPipeline
│   ├── baselines/           # Majority baseline & TF-IDF baseline
│   ├── escalation/          # Safety-critical rule-based escalation policy
│   ├── evaluation/          # Metrics calculation & GroqJudge rubric
│   ├── generation/          # Groq LLM reply generator with exponential backoff
│   ├── intents/             # Taxonomy rules & LogReg classifier
│   └── retrieval/           # Evidence retriever & vector indexer
└── tests/                   # 18 unit tests (100% passing)
```

---

## 🛡️ Decisions & Design Choices

See [DECISIONS.md](DECISIONS.md) for detailed reasoning on 13 non-obvious engineering decisions, including:
- Why usable pairs was prioritized over raw tweet volume for brand selection.
- Why escalation is strictly rule-based rather than a black-box model.
- Why financial/billing disputes always escalate regardless of intent confidence.
