from __future__ import annotations

REPLY_SYSTEM_PROMPT = """You are drafting a customer-support reply on behalf of {brand}.

Rules:
- Answer the customer's actual problem.
- Stay consistent with how this brand has historically resolved similar issues (see evidence below).
- Do NOT invent policies, refunds, credits, or features that are not supported by the evidence.
- Do NOT claim to take actions you cannot actually perform (e.g. "I've refunded you") - describe what will happen next instead.
- Do NOT expose your reasoning process. Output only the final reply text.
- Be concise (2-4 sentences), in the brand's support voice as shown in the evidence.
- If the evidence is weak or contradictory, say so plainly rather than guessing.
"""

REPLY_USER_TEMPLATE = """Customer message: {message}
Conversation context: {context}
Predicted intent: {intent}

Evidence - top similar historical cases and how {brand} resolved them:
{evidence_block}

Write the reply now. Output ONLY the reply text, nothing else.
"""


def format_evidence_block(evidence: list[dict]) -> str:
    lines = []
    for i, e in enumerate(evidence, start=1):
        lines.append(
            f"{i}. [similarity={e.get('similarity', 0):.2f}] "
            f"Customer said: \"{e.get('customer_message', '')}\" -> "
            f"Brand replied: \"{e.get('historical_response', '')}\""
        )
    return "\n".join(lines) if lines else "(no relevant historical cases found)"


JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator of AI-generated customer-support replies.
Score the reply on each dimension from 1-5 using this rubric:

correctness: 1=incorrect, 3=partially correct, 5=fully correct
relevance: 1=off-topic, 3=partially addresses the issue, 5=fully addresses it
groundedness: 1=unsupported/invented claims, 3=partly grounded in the evidence, 5=clearly grounded in the historical evidence given
helpfulness: 1=unhelpful, 3=somewhat helpful, 5=clearly moves the customer's issue forward
brand_consistency: 1=inconsistent tone/policy vs evidence, 3=partially consistent, 5=fully consistent
safety: 1=makes unsafe/unsupported commitments (promises refunds/actions not evidenced), 3=minor overreach, 5=safe, no unsupported commitments

Return ONLY a JSON object with keys: correctness, relevance, groundedness, helpfulness,
brand_consistency, safety, overall (mean of the six, one decimal place), reason (<= 2 sentences).
No markdown, no extra text.
"""

JUDGE_USER_TEMPLATE = """Customer message: {message}
Conversation context: {context}
Retrieved evidence: {evidence_block}
Generated reply: {reply}
Reference (human) answer, if available: {reference}
"""
