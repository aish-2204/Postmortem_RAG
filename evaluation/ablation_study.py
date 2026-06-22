"""
Ablation study: compare all 4 retrieval strategies on the same QA pairs.

Runs ragas_evaluator for each strategy in sequence and writes a comparison
table to evaluation/results/ablation_results.md.

Usage:
    python -m evaluation.ablation_study              # all 4 strategies, full dataset
    python -m evaluation.ablation_study --limit 20   # quick test on 20 pairs
    python -m evaluation.ablation_study --skip dense sparse  # skip already-run strategies
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation" / "results"

_STRATEGIES = ["dense", "sparse", "hybrid", "hybrid_rerank"]

_THRESHOLDS = {
    "context_recall":    0.70,
    "context_precision": 0.75,
    "faithfulness":      0.82,
    "answer_relevancy":  0.78,
}


def run(limit: int | None = None, skip: list[str] | None = None) -> None:
    from evaluation.ragas_evaluator import evaluate

    skip = skip or []
    all_results = {}

    # Load any previously run strategy results
    for strategy in _STRATEGIES:
        cached = RESULTS_DIR / f"eval_{strategy}.json"
        if cached.exists() and strategy in skip:
            print(f"Loading cached results for: {strategy}")
            data = json.loads(cached.read_text())
            all_results[strategy] = data

    # Run missing strategies
    for strategy in _STRATEGIES:
        if strategy in skip and strategy in all_results:
            continue
        print(f"\nRunning strategy: {strategy}")
        result = evaluate(strategy=strategy, top_k=5, limit=limit)
        all_results[strategy] = result

    _write_markdown(all_results, limit)


def _gate(value: float, metric: str) -> str:
    threshold = _THRESHOLDS[metric]
    return f"{value:.4f} {'✓' if value >= threshold else '✗'}"


def _write_markdown(results: dict, limit: int | None) -> None:
    lines = []
    n = list(results.values())[0]["n_pairs"] if results else 0
    dataset_note = f"{n} QA pairs" + (f" (limit={limit})" if limit else "")

    lines.append(f"# Retrieval Ablation Study")
    lines.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d')}  ")
    lines.append(f"**Dataset:** {dataset_note}  ")
    lines.append(f"**Thresholds:** recall≥0.70, precision≥0.75, faithfulness≥0.82, relevancy≥0.78\n")

    # Main comparison table
    lines.append("## Results\n")
    header = "| Strategy         | Context Recall | Context Precision | Faithfulness | Answer Relevancy | CI Gate |"
    sep    = "|------------------|---------------|-------------------|--------------|-----------------|---------|"
    lines.append(header)
    lines.append(sep)

    for strategy in _STRATEGIES:
        if strategy not in results:
            continue
        r = results[strategy]
        gate_pass = all(
            r[m] >= t for m, t in _THRESHOLDS.items()
        )
        row = (
            f"| {strategy:<16} "
            f"| {_gate(r['context_recall'],    'context_recall'):>13} "
            f"| {_gate(r['context_precision'], 'context_precision'):>17} "
            f"| {_gate(r['faithfulness'],      'faithfulness'):>12} "
            f"| {_gate(r['answer_relevancy'],  'answer_relevancy'):>15} "
            f"| {'PASS ✓' if gate_pass else 'FAIL ✗'} |"
        )
        lines.append(row)

    # Winner analysis
    lines.append("\n## Key Findings\n")
    if len(results) == 4:
        best_recall    = max(results, key=lambda s: results[s]["context_recall"])
        best_precision = max(results, key=lambda s: results[s]["context_precision"])
        best_faith     = max(results, key=lambda s: results[s]["faithfulness"])
        best_relev     = max(results, key=lambda s: results[s]["answer_relevancy"])

        lines.append(f"- **Best context recall:**    `{best_recall}` ({results[best_recall]['context_recall']:.4f})")
        lines.append(f"- **Best context precision:** `{best_precision}` ({results[best_precision]['context_precision']:.4f})")
        lines.append(f"- **Best faithfulness:**      `{best_faith}` ({results[best_faith]['faithfulness']:.4f})")
        lines.append(f"- **Best answer relevancy:**  `{best_relev}` ({results[best_relev]['answer_relevancy']:.4f})")

        # Delta: hybrid_rerank vs dense
        if "hybrid_rerank" in results and "dense" in results:
            delta_recall = results["hybrid_rerank"]["context_recall"] - results["dense"]["context_recall"]
            delta_prec   = results["hybrid_rerank"]["context_precision"] - results["dense"]["context_precision"]
            lines.append(f"\n**Hybrid+Rerank vs Dense only:**")
            lines.append(f"- Context recall delta:    {delta_recall:+.4f}")
            lines.append(f"- Context precision delta: {delta_prec:+.4f}")

    # Per question-type breakdown (if details available)
    lines.append("\n## Breakdown by Question Type\n")
    lines.append("| Question Type    | Strategy         | Context Recall | Context Precision |")
    lines.append("|------------------|------------------|---------------|-------------------|")
    for strategy in _STRATEGIES:
        if strategy not in results or "details" not in results[strategy]:
            continue
        details = results[strategy]["details"]
        for qt in ["root_cause", "remediation", "lessons_learned"]:
            subset = [d for d in details if d.get("question_type") == qt]
            if not subset:
                continue
            avg_recall = sum(d["context_recall"]    for d in subset) / len(subset)
            avg_prec   = sum(d["context_precision"] for d in subset) / len(subset)
            lines.append(
                f"| {qt:<16} | {strategy:<16} | {avg_recall:.4f}        | {avg_prec:.4f}            |"
            )

    out_path = RESULTS_DIR / "ablation_results.md"
    out_path.write_text("\n".join(lines))
    print(f"\nAblation results saved to {out_path}")

    # Print to console too
    print("\n" + "\n".join(lines[:20]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N pairs per strategy")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=_STRATEGIES,
                        help="Skip these strategies (use cached results)")
    args = parser.parse_args()
    run(limit=args.limit, skip=args.skip)
