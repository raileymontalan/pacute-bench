"""
PACUTE: Phonology, Affix, and Character-level Understanding of Tokens Evaluation

Full morphological understanding benchmark covering:
- Composition (MCQ + generative)
- Manipulation (MCQ + generative)
- Syllabification (MCQ + generative)
- Morphological Extraction (MCQ + generative)
- Morphological Production (MCQ + generative)
"""
import json
import random
from pathlib import Path


def load_pacute(
    split: str = "test",
    categories: list = None,
    format: str = "mcq",
    data_dir: str = "data/benchmarks",
    **kwargs,
):
    """
    Load PACUTE benchmark.

    Args:
        split: Unused (all data treated as test).
        categories: List of PACUTE categories to include. Defaults to all five:
            ['composition', 'manipulation', 'syllabification',
             'morphological_extraction', 'morphological_production'].
        format: ``"mcq"`` or ``"gen"``.
        data_dir: Directory containing the benchmark JSONL files.

    Yields:
        (prefix, ground_truth, false_options, sample_id, subcategory)
    """
    benchmarks_dir = Path(data_dir)

    if categories is None:
        categories = ["composition", "manipulation", "syllabification",
                      "morphological_extraction", "morphological_production"]

    tasks = []
    category_counts = {}

    suffix = "gen" if format == "gen" else "mcq"

    for category in categories:
        data_file = benchmarks_dir / f"{category}_{suffix}.jsonl"
        if not data_file.exists():
            print(f"Warning: PACUTE file not found: {data_file}")
            continue

        count = 0
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                task = json.loads(line)
                task["_category"] = category
                tasks.append(task)
                count += 1
        category_counts[category] = count

    total = len(tasks)
    print(f"PACUTE ({format}): Loaded {total} tasks across {len(category_counts)} categories:")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count} tasks")

    indices = list(range(len(tasks)))
    random.shuffle(indices)

    for i in indices:
        task = tasks[i]
        prompt_data = task["prompts"][0]
        prefix = prompt_data["text_en"]
        sample_id = task.get("id", f"pacute_{format}_{i:05d}")

        if format == "gen":
            ground_truth = task["label"]
            false_options = []
        else:
            mcq_options = prompt_data["mcq_options"]
            ground_truth = mcq_options["correct"]
            false_options = [
                v for k, v in sorted(mcq_options.items()) if k.startswith("incorrect")
            ]

        yield prefix, ground_truth, false_options, sample_id, task.get("subcategory")
