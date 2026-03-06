"""
CUTE: Character Understanding Test Evaluation

Tests character-level understanding tasks (spelling, insertion, deletion, etc.)
Based on Edman et al. (2024) "CUTE: Measuring LLMs' Understanding of Their Tokens"

Dataset: https://huggingface.co/datasets/leukas/cute
"""
import json
import random
from pathlib import Path


def load_cute(
    split: str = "test",
    task_types: list = None,
    max_per_task: int = 100,
    data_dir: str = "data/benchmarks",
    **kwargs,
):
    """
    Load CUTE benchmark from a local JSONL file.

    Args:
        split: Unused (all data treated as test).
        task_types: List of task types to include. Defaults to all 14.
        max_per_task: Maximum examples per task type (default: 100).
        data_dir: Directory containing the benchmark JSONL files.

    Yields:
        (prefix, ground_truth, false_options, sample_id, task_type)
    """
    jsonl_path = Path(data_dir) / "cute_gen.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"CUTE benchmark file not found: {jsonl_path}\n"
            "Generate it with: python scripts/generate_benchmarks.py --benchmarks cute"
        )

    all_task_types = [
        "spell", "spell_inverse", "contains_char", "contains_word",
        "orth", "sem", "ins_char", "ins_word", "del_char", "del_word",
        "sub_char", "sub_word", "swap_char", "swap_word",
    ]
    if task_types is None:
        task_types = all_task_types

    all_samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            all_samples.append(json.loads(line.strip()))

    tasks = []
    task_counts: dict = {}
    for sample in all_samples:
        tt = sample.get("task_type", "unknown")
        if tt in task_types:
            task_counts[tt] = task_counts.get(tt, 0) + 1
            tasks.append(sample)

    print(
        f"CUTE (gen): Loaded {len(tasks)} tasks "
        f"({len(task_counts)} task types) from {jsonl_path}"
    )

    random.shuffle(tasks)

    for i, task in enumerate(tasks):
        yield (
            task["question"],
            task["answer"],
            [],  # generative — no MCQ options
            task.get("id", f"cute_gen_{i:05d}"),
            task.get("task_type"),
        )
