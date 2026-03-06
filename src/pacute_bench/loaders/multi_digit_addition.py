"""
Multi-digit Addition — simple arithmetic tasks.

Format: "X+Y=" → "Z"
"""
import json
import random
from pathlib import Path


def load_multi_digit_addition(
    format: str = "gen",
    max_samples: int = 1000,
    data_dir: str = "data/benchmarks",
    **kwargs,
):
    """
    Load multi-digit addition benchmark from a local JSONL file.

    Args:
        format: ``"gen"`` (default) or ``"mcq"``.
        max_samples: Maximum number of examples to load (default: 1000).
        data_dir: Directory containing benchmark JSONL files.

    Yields:
        (prefix, ground_truth, false_options, sample_id, None)
    """
    filepath = Path(data_dir) / f"multi_digit_addition_{format}.jsonl"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Multi-digit addition benchmark file not found: {filepath}\n"
            "Generate it with: python scripts/generate_benchmarks.py --benchmarks multi-digit-addition"
        )

    samples = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
            if max_samples is not None and len(samples) >= max_samples:
                break

    print(f"Multi-digit Addition ({format}): Loaded {len(samples)} examples from {filepath}")
    random.shuffle(samples)

    for i, sample in enumerate(samples):
        sample_id = sample.get("id", f"multi_digit_addition_{format}_{i:05d}")

        if format == "mcq":
            options = sample["options"]
            ground_truth = options[0]
            false_options = options[1:]
        else:
            ground_truth = sample["answer"]
            false_options = []

        yield sample["question"], ground_truth, false_options, sample_id, None
