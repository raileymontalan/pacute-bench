"""
LangGame — word-property language games (longest, shortest, starts-with, etc.)
"""
import json
import random
from pathlib import Path


def load_langgame(
    format: str = "mcq",
    data_dir: str = "data/benchmarks",
    **kwargs,
):
    """
    Load LangGame benchmark from a local JSONL file.

    Args:
        format: ``"mcq"`` or ``"gen"``.
        data_dir: Directory containing benchmark JSONL files.

    Yields:
        (prefix, ground_truth, false_options, sample_id, None)
    """
    filepath = Path(data_dir) / f"langgame_{format}.jsonl"

    if not filepath.exists():
        raise FileNotFoundError(
            f"LangGame benchmark file not found: {filepath}\n"
            "Generate it with: python scripts/generate_benchmarks.py --benchmarks langgame"
        )

    samples = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    print(f"LangGame ({format}): Loaded {len(samples)} examples from {filepath}")
    random.shuffle(samples)

    for i, sample in enumerate(samples):
        sample_id = sample.get("id", f"langgame_{format}_{i:05d}")

        if format == "mcq":
            options = sample["options"]
            ground_truth = options[0]
            false_options = options[1:]
        else:
            ground_truth = sample["answer"]
            false_options = []

        yield sample["question"], ground_truth, false_options, sample_id, None
