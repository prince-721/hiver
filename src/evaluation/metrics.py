from __future__ import annotations

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def _fallback_intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / n if n else 0.0

    label_to_idx = {lb: i for i, lb in enumerate(labels)}
    cm = [[0] * len(labels) for _ in range(len(labels))]
    for yt, yp in zip(y_true, y_pred):
        cm[label_to_idx[yt]][label_to_idx[yp]] += 1

    per_class = {}
    f1_list = []
    for lb in labels:
        idx = label_to_idx[lb]
        tp = cm[idx][idx]
        fp = sum(cm[r][idx] for r in range(len(labels)) if r != idx)
        fn = sum(cm[idx][c] for c in range(len(labels)) if c != idx)
        support = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

        per_class[lb] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1-score": round(f1, 4),
            "support": support,
        }

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_report": per_class,
        "confusion_matrix": cm,
        "labels": labels,
    }


def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    if not _SKLEARN_AVAILABLE:
        return _fallback_intent_metrics(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class_report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def escalation_metrics(y_true_escalate: list[bool], y_pred_escalate: list[bool]) -> dict:
    n = len(y_true_escalate)
    if _SKLEARN_AVAILABLE:
        precision = float(precision_score(y_true_escalate, y_pred_escalate, zero_division=0))
        recall = float(recall_score(y_true_escalate, y_pred_escalate, zero_division=0))
        f1 = float(f1_score(y_true_escalate, y_pred_escalate, zero_division=0))
    else:
        tp = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if t and p)
        fp = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if not t and p)
        fn = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if t and not p)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # false auto-handle: gold says ESCALATE but system said AUTO_HANDLE - the
    # dangerous error direction (an unsafe reply reached a customer).
    false_auto_handle = sum(
        1 for t, p in zip(y_true_escalate, y_pred_escalate) if t and not p
    )
    automation_coverage = sum(1 for p in y_pred_escalate if not p) / n if n else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_auto_handle_rate": round(false_auto_handle / n, 4) if n else 0.0,
        "automation_coverage": round(automation_coverage, 4),
        "n": n,
    }


def retrieval_recall_at_k(hits_per_query: list[list[str]], relevant_per_query: list[set[str]], k: int) -> float:
    """hits_per_query: ranked list of retrieved ids per query (already truncated or not).
    relevant_per_query: set of relevant ids per query (here: the single true historical match, if known).
    """
    n = len(hits_per_query)
    if n == 0:
        return 0.0
    hits = 0
    for retrieved, relevant in zip(hits_per_query, relevant_per_query):
        if set(retrieved[:k]) & relevant:
            hits += 1
    return hits / n


def mrr(hits_per_query: list[list[str]], relevant_per_query: list[set[str]]) -> float:
    n = len(hits_per_query)
    if n == 0:
        return 0.0
    total = 0.0
    for retrieved, relevant in zip(hits_per_query, relevant_per_query):
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                total += 1.0 / rank
                break
    return total / n
