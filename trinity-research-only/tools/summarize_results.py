"""Recompute descriptive arithmetic from transcribed aggregate counts only.

This script does not load a model, access a network, or independently verify
historical grades. Result rows describe different stages, not pooled trials.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    source = Path(__file__).resolve().parents[1] / "results" / "reported_metrics.json"
    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)
    print(data["evidence_status"])
    print()
    for name, row in data.items():
        if not isinstance(row, dict) or "mutually_scorable" not in row:
            continue
        total = row["mutually_scorable"]
        baseline, candidate = row["baseline_correct"], row["candidate_correct"]
        print(f"{name}: baseline {baseline}/{total} ({baseline/total:.1%}); "
              f"candidate {candidate}/{total} ({candidate/total:.1%})")
        if "baseline_reported_tokens" in row:
            before, after = row["baseline_reported_tokens"], row["candidate_reported_tokens"]
            print(f"  Reported token change: {after-before:+,} ({(after-before)/before:+.2%})")
    print("\nv5: optimization-set measurements only")
    for i, fitnesses in enumerate(data["v5_training"]["generation_candidate_fitnesses"], 1):
        print(f"  Generation {i}: mean={sum(fitnesses)/len(fitnesses):.3f}; best={max(fitnesses):.2f}")


if __name__ == "__main__":
    main()
