"""
Hierarchical benchmark — diagnostic tasks across 6 compositional levels.

Levels:
  0  Character Recognition
  1  Character Manipulation
  2  Morpheme Decomposition
  3  Morpheme Manipulation
  4  Morpheme Composition
  5  Complex Morphological Reasoning
"""
import json
import random
from pathlib import Path


def load_hierarchical(
    format: str = "mcq",
    data_dir: str = "data/benchmarks",
    **kwargs,
):
    """
    Load hierarchical benchmark from a local JSONL file.

    Args:
        format: ``"mcq"`` or ``"gen"``.
        data_dir: Directory containing benchmark JSONL files.

    Yields:
        (prefix, ground_truth, false_options, sample_id, category)
    """
    filepath = Path(data_dir) / f"hierarchical_{format}.jsonl"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Hierarchical benchmark file not found: {filepath}\n"
            "Generate it with: python scripts/generate_benchmarks.py --benchmarks hierarchical"
        )

    tasks = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line.strip()))

    print(f"Hierarchical ({format}): Loaded {len(tasks)} tasks from {filepath}")
    random.shuffle(tasks)

    for i, task in enumerate(tasks):
        prefix = task.get("prompt_tl", task.get("question", ""))
        sample_id = task.get("id", f"hierarchical_{format}_{i:05d}")

        if format == "mcq":
            options = task["options"]
            ground_truth = options[0]
            false_options = options[1:]
        else:
            ground_truth = task["answer"]
            false_options = []

        cat_parts = [task.get("category", ""), task.get("subcategory", "")]
        category = "/".join(p for p in cat_parts if p) or None

        yield prefix, ground_truth, false_options, sample_id, category
