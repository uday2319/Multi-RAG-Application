"""
Evaluate the RAG pipeline against eval/golden_set.json.

Retrieval (page level, no LLM needed):
  hit@k        - a gold (paper, page) is in the top k results
  mrr          - 1 / rank of the first gold result
  paper_cov@k  - cross-paper questions: share of required papers in the top k

Generation (uses Ollama, skip with --retrieval-only):
  answer_score     - share of required keyword groups found in the answer
  answer_correct   - all keyword groups found
  cited            - answer contains at least one [file.pdf, p.N] citation
  citation_gold    - share of citations that point to a gold page
  citation_grounded- share of citations that point to a page actually retrieved
  refused          - answer says "not found"; correct for unanswerable questions

Usage (from the project root):
  python eval/run_eval.py --label baseline
  python eval/run_eval.py --label baseline --retrieval-only
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configure import TOP_K  # noqa: E402
from main import create_rag  # noqa: E402

GOLDEN_SET = ROOT / "eval" / "golden_set.json"
RESULTS_DIR = ROOT / "eval" / "results"

K_VALUES = [1, 3, 5, 10]

CITATION_RE = re.compile(
    r"\[\s*([^\[\],]+?\.pdf)\s*,\s*p(?:age|\.)?\s*(\d+)\s*\]",
    re.IGNORECASE
)

REFUSAL_RE = re.compile(r"not found", re.IGNORECASE)


def page_key(item):
    return (item["source"], int(item["page"]))


def retrieval_metrics(results, item):

    gold = {page_key(g) for g in item["gold"]}
    keys = [page_key(r) for r in results]

    metrics = {}

    for k in K_VALUES:
        metrics[f"hit@{k}"] = float(any(key in gold for key in keys[:k]))

    rank = next(
        (i for i, key in enumerate(keys, start=1) if key in gold),
        None
    )
    metrics["mrr"] = 1.0 / rank if rank else 0.0

    if item.get("required_papers"):
        required = set(item["required_papers"])
        for k in K_VALUES:
            found = {r["source"] for r in results[:k]} & required
            metrics[f"paper_cov@{k}"] = len(found) / len(required)

    return metrics


def answer_metrics(answer, item, used_results):

    text = answer.lower()

    refused = bool(REFUSAL_RE.search(answer))

    citations = {
        (source.strip(), int(page))
        for source, page in CITATION_RE.findall(answer)
    }

    gold = {page_key(g) for g in item["gold"]}
    retrieved = {page_key(r) for r in used_results}

    metrics = {
        "refused": float(refused),
        "cited": float(bool(citations)),
        "num_citations": len(citations),
    }

    if citations:
        metrics["citation_grounded"] = (
            len(citations & retrieved) / len(citations)
        )
        if gold:
            metrics["citation_gold"] = len(citations & gold) / len(citations)

    groups = item["answer_keywords"]

    if groups:
        matched = sum(
            any(alt.lower() in text for alt in group)
            for group in groups
        )
        metrics["answer_score"] = matched / len(groups)
        metrics["answer_correct"] = float(matched == len(groups))

    return metrics


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize(rows):

    def subset(types):
        return [r for r in rows if r["type"] in types]

    def avg(rs, key):
        return mean(r["metrics"][key] for r in rs if key in r["metrics"])

    answerable = subset({"single", "cross"})
    cross = subset({"cross"})
    unanswerable = subset({"unanswerable"})

    summary = {"counts": {
        "total": len(rows),
        "answerable": len(answerable),
        "cross": len(cross),
        "unanswerable": len(unanswerable),
    }}

    retrieval = {f"hit@{k}": avg(answerable, f"hit@{k}") for k in K_VALUES}
    retrieval["mrr"] = avg(answerable, "mrr")
    for k in K_VALUES:
        retrieval[f"cross_paper_cov@{k}"] = avg(cross, f"paper_cov@{k}")
    summary["retrieval"] = retrieval

    def generation(prefix):
        refusal = avg(unanswerable, f"{prefix}refused")
        return {
            "answer_score": avg(answerable, f"{prefix}answer_score"),
            "answer_correct": avg(answerable, f"{prefix}answer_correct"),
            "cross_answer_correct": avg(cross, f"{prefix}answer_correct"),
            "cited_rate": avg(answerable, f"{prefix}cited"),
            "citation_gold": avg(answerable, f"{prefix}citation_gold"),
            "citation_grounded": avg(answerable, f"{prefix}citation_grounded"),
            "false_refusal_rate": avg(answerable, f"{prefix}refused"),
            "correct_refusal_rate": refusal,
            "hallucination_rate_unanswerable": (
                1 - refusal if refusal is not None else None
            ),
        }

    if rows and "refused" in rows[0]["metrics"]:
        summary["generation"] = generation("")
        summary["generation"]["avg_latency_s"] = mean(r["latency_s"] for r in rows)

    if rows and "raw_refused" in rows[0]["metrics"]:
        summary["generation_raw"] = generation("raw_")

        # The guard's trade-off in plain counts (answerable questions)
        summary["guard_effect"] = {
            "correct_answers_removed": sum(
                1 for r in answerable
                if r["metrics"].get("raw_answer_correct") == 1
                and r["metrics"].get("answer_correct") == 0
            ),
            "wrong_answers_blocked": sum(
                1 for r in answerable
                if r["metrics"].get("raw_answer_correct") == 0
                and r["metrics"].get("raw_refused") == 0
                and r["metrics"].get("refused") == 1
            ),
            "unanswerable_hallucinations_blocked": sum(
                1 for r in unanswerable
                if r["metrics"].get("raw_refused") == 0
                and r["metrics"].get("refused") == 1
            ),
        }

    return summary


def format_summary(summary, label):

    def fmt(value):
        return "-" if value is None else f"{value:.3f}"

    lines = [f"## {label}", ""]
    lines.append(f"Questions: {summary['counts']}")
    lines.append("")
    lines.append("| Retrieval metric | Value |")
    lines.append("|---|---|")
    for key, value in summary["retrieval"].items():
        lines.append(f"| {key} | {fmt(value)} |")

    if "generation" in summary and "generation_raw" in summary:
        lines.append("")
        lines.append("| Generation metric | Raw LLM | With guard |")
        lines.append("|---|---|---|")
        for key, value in summary["generation"].items():
            raw = summary["generation_raw"].get(key)
            lines.append(f"| {key} | {fmt(raw)} | {fmt(value)} |")

        if "guard_effect" in summary:
            lines.append("")
            lines.append("| Guard effect | Count |")
            lines.append("|---|---|")
            for key, value in summary["guard_effect"].items():
                lines.append(f"| {key} | {value} |")

    elif "generation" in summary:
        lines.append("")
        lines.append("| Generation metric | Value |")
        lines.append("|---|---|")
        for key, value in summary["generation"].items():
            lines.append(f"| {key} | {fmt(value)} |")

    return "\n".join(lines)


def apply_guard(row, question, raw_answer, used):
    """Run the hallucination guard on a raw answer; store the result in row."""

    from LLM.generator import NOT_FOUND
    from LLM.guard import evidence_gate, verify_answer

    gate_passed, best_score = evidence_gate(used)

    if gate_passed:
        check = verify_answer(question, raw_answer, used)
        row["answer"] = check["answer"]
        row["guard_reason"] = check["reason"]
    else:
        row["answer"] = NOT_FOUND
        row["guard_reason"] = "weak_evidence"

    row["best_rerank_score"] = best_score


def reguard(path, items, retriever):

    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in items}

    for row in data["rows"]:

        if "raw_answer" not in row:
            continue

        item = by_id[row["id"]]

        results = retriever.retrieve(item["question"], top_k=max(K_VALUES))
        used = results[:data["top_k"]]

        apply_guard(row, item["question"], row["raw_answer"], used)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{stamp}_{data['label']}_reguard.json"

    data["label"] = data["label"] + "_reguard"
    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    rescore(out_path, items)


def rescore(path, items):

    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in items}

    for row in data["rows"]:

        item = by_id[row["id"]]
        results = row["retrieved"]

        metrics = retrieval_metrics(results, item)

        used = results[:data["top_k"]]

        if "answer" in row:
            metrics.update(answer_metrics(row["answer"], item, used))

        if "raw_answer" in row:
            metrics.update({
                f"raw_{key}": value
                for key, value in answer_metrics(row["raw_answer"], item, used).items()
            })

        row["metrics"] = metrics

    data["summary"] = summarize(data["rows"])

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(format_summary(data["summary"], data["label"] + " (rescored)"))
    print(f"\nUpdated: {path}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mode", choices=["dense", "hybrid"],
        help="Override RETRIEVAL_MODE from configure.py"
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Disable the cross-encoder reranker"
    )
    parser.add_argument(
        "--routing", choices=["off", "multi", "all"],
        help="Override PAPER_ROUTING from configure.py"
    )
    parser.add_argument(
        "--reguard",
        help="Re-apply the (changed) guard to the raw answers in a "
             "saved results file: re-runs retrieval, not the LLM"
    )
    parser.add_argument(
        "--rescore",
        help="Re-score a saved results file against the current "
             "golden set without re-running retrieval or the LLM"
    )
    args = parser.parse_args()

    items = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))

    if args.rescore:
        rescore(Path(args.rescore), items)
        return

    rag_kwargs = {}
    if args.mode:
        rag_kwargs["mode"] = args.mode
    if args.no_rerank:
        rag_kwargs["reranker_model"] = None
    if args.routing:
        rag_kwargs["routing"] = args.routing
    if args.limit:
        items = items[:args.limit]

    retriever = create_rag(**rag_kwargs)

    if args.reguard:
        reguard(Path(args.reguard), items, retriever)
        return

    generate_answer = None
    if not args.retrieval_only:
        from LLM.generator import generate_answer

    rows = []

    for i, item in enumerate(items, start=1):

        start = time.time()

        results = retriever.retrieve(
            item["question"],
            top_k=max(K_VALUES)
        )

        # The generator sees exactly what the live system sees
        used = results[:TOP_K]

        metrics = retrieval_metrics(results, item)

        row = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "retrieved": [
                {
                    "source": r["source"],
                    "page": r["page"],
                    "score": round(r["score"], 4),
                    "dense_score": (
                        round(r["dense_score"], 4)
                        if r.get("dense_score") is not None else None
                    ),
                    "rerank_score": (
                        round(r["rerank_score"], 4)
                        if r.get("rerank_score") is not None else None
                    )
                }
                for r in results
            ],
        }

        if generate_answer:

            # Raw LLM answer (no guard). Generated even when the
            # evidence gate would refuse, so the guard's effect can
            # be measured on the same run.
            raw_answer = generate_answer(item["question"], used)
            row["raw_answer"] = raw_answer

            apply_guard(row, item["question"], raw_answer, used)

            metrics.update(answer_metrics(row["answer"], item, used))
            metrics.update({
                f"raw_{key}": value
                for key, value in answer_metrics(raw_answer, item, used).items()
            })

        row["metrics"] = metrics
        row["latency_s"] = round(time.time() - start, 2)
        rows.append(row)

        status = (
            f"hit@{TOP_K}={metrics[f'hit@{TOP_K}']:.0f}"
            if item["gold"] else "unanswerable"
        )
        if "answer_correct" in metrics:
            status += f" correct={metrics['answer_correct']:.0f}"
        if "refused" in metrics:
            status += f" refused={metrics['refused']:.0f}"

        print(f"[{i}/{len(items)}] {item['id']}: {status}", flush=True)

    summary = summarize(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{stamp}_{args.label}.json"

    out_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "timestamp": stamp,
                "top_k": TOP_K,
                "summary": summary,
                "rows": rows
            },
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    report = format_summary(summary, args.label)
    print("\n" + report)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
