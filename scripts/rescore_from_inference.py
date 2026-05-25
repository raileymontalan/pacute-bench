"""
Re-aggregate evaluation_results_*.json from existing inference JSONL files.

Usage:
    python scripts/rescore_from_inference.py                        # all models
    python scripts/rescore_from_inference.py gemma-4-e2b-it         # one model
    python scripts/rescore_from_inference.py gemma-4-e2b-it gpt-oss-20b  # several

Writes a new evaluation_results_<timestamp>.json alongside the existing ones.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results-202605"


# ── Scoring helpers (mirror base.py logic, no torch dependency) ───────────────

def _softmax(vals: list[float]) -> list[float]:
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]


def _score_mcq(samples: list[dict]) -> dict:
    n = len(samples)
    correct = [r for r in samples if r["is_correct"]]
    acc = len(correct) / n
    tp = len(correct)
    fn = n - tp
    precision = tp / (tp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    path_conf_vals = []
    for r in samples:
        lps = r.get("logprobs")
        if lps:
            path_conf_vals.append(_softmax(lps)[0])
    path_confidence = sum(path_conf_vals) / len(path_conf_vals) if path_conf_vals else 0.0

    num_opts = len(samples[0]["options"]) if samples[0].get("options") else 4
    norm_acc = (acc * num_opts - 1) / (num_opts - 1)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "path_confidence": path_confidence,
        "normalized_accuracy": norm_acc,
        "num_options": num_opts,
        "num_samples": n,
        "format": "mcq",
    }


def _score_gen(samples: list[dict]) -> dict:
    n = len(samples)
    return {
        "exact_match": sum(r["exact_match"] for r in samples) / n,
        "contains_match": sum(r["contains_match"] for r in samples) / n,
        "prefix_match": sum(r["prefix_match"] for r in samples) / n,
        "num_samples": n,
        "format": "gen",
    }


def _by_category_mcq(samples: list[dict]) -> dict:
    groups: dict = {}
    for r in samples:
        groups.setdefault(r.get("category") or "__all__", []).append(r)
    out = {}
    for cat, items in sorted(groups.items()):
        s = _score_mcq(items)
        out[cat] = {k: round(v, 4) for k, v in s.items() if k not in ("num_options", "format")}
        out[cat]["num_samples"] = s["num_samples"]
    return out


def _by_category_gen(samples: list[dict]) -> dict:
    groups: dict = {}
    for r in samples:
        groups.setdefault(r.get("category") or "__all__", []).append(r)
    out = {}
    for cat, items in sorted(groups.items()):
        n = len(items)
        out[cat] = {
            "num_samples": n,
            "exact_match": round(sum(r["exact_match"] for r in items) / n, 4),
            "contains_match": round(sum(r["contains_match"] for r in items) / n, 4),
            "prefix_match": round(sum(r["prefix_match"] for r in items) / n, 4),
        }
    return out


# ── Per-model rescoring ───────────────────────────────────────────────────────

def rescore_model(model_dir: Path) -> None:
    inference_dir = model_dir / "inference"
    if not inference_dir.exists():
        print(f"  SKIP {model_dir.name}: no inference/ dir")
        return

    # Pull metadata from the most recent non-empty result file
    meta = {"hf_model_name": model_dir.name, "model_type": "?", "thinking": False}
    for f in sorted(model_dir.glob("evaluation_results_*.json"), reverse=True):
        d = json.loads(f.read_text())
        if d.get("benchmarks") is not None:
            meta = {k: d[k] for k in ("hf_model_name", "model_type", "thinking") if k in d}
            break

    benchmarks: dict = {}
    for jsonl in sorted(inference_dir.glob("*.jsonl")):
        bench_name = jsonl.stem
        samples = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        if not samples:
            continue

        is_mcq = "is_correct" in samples[0] and "logprobs" in samples[0]
        if is_mcq:
            scores = _score_mcq(samples)
            scores["by_category"] = _by_category_mcq(samples)
        else:
            scores = _score_gen(samples)
            scores["by_category"] = _by_category_gen(samples)

        benchmarks[bench_name] = scores

    if not benchmarks:
        print(f"  SKIP {model_dir.name}: no inference JSONL files")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {**meta, "timestamp": timestamp, "benchmarks": benchmarks}
    out_path = model_dir / f"evaluation_results_{timestamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  OK {model_dir.name}: {len(benchmarks)} benchmarks → {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="model dir names; omit for all")
    args = parser.parse_args()

    if args.models:
        dirs = [RESULTS_DIR / m for m in args.models]
    else:
        dirs = sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())

    for d in dirs:
        if not d.exists():
            print(f"  NOT FOUND: {d}")
            continue
        rescore_model(d)


if __name__ == "__main__":
    main()
