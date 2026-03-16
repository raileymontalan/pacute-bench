"""
BaseEvaluator — shared state, metrics, answer extraction, and result persistence.

All concrete evaluators (VLLMEvaluator, CommercialEvaluator) inherit from this.
"""

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ..loaders import load_benchmark, BENCHMARK_REGISTRY  # noqa: F401

# ---------------------------------------------------------------------------
# Benchmark format registry
# ---------------------------------------------------------------------------

#: Maps every registered benchmark name to "mcq" or "gen".
BENCHMARK_FORMATS: dict = {
    "cute":     "gen",
    "cute-gen": "gen",

    "hierarchical":     "mcq",
    "hierarchical-mcq": "mcq",
    "hierarchical-gen": "gen",

    "langgame":     "mcq",
    "langgame-mcq": "mcq",
    "langgame-gen": "gen",

    "multi-digit-addition":     "gen",
    "multi-digit-addition-gen": "gen",
    "multi-digit-addition-mcq": "mcq",

    "pacute":     "mcq",
    "pacute-mcq": "mcq",
    "pacute-gen": "gen",

    "pacute-affixation":          "mcq",
    "pacute-affixation-mcq":      "mcq",
    "pacute-affixation-gen":      "gen",
    "pacute-composition":         "mcq",
    "pacute-composition-mcq":     "mcq",
    "pacute-composition-gen":     "gen",
    "pacute-manipulation":        "mcq",
    "pacute-manipulation-mcq":    "mcq",
    "pacute-manipulation-gen":    "gen",
    "pacute-syllabification":     "mcq",
    "pacute-syllabification-mcq": "mcq",
    "pacute-syllabification-gen": "gen",
}


class BaseEvaluator(ABC):
    """
    Abstract base for all pacute-bench evaluators.

    Provides:
    - Common ``__init__`` that stores shared configuration.
    - Answer extraction (``_extract_answer``) — deduplicated from all subclasses.
    - MCQ metrics (``_calc_metrics``) and per-category breakdowns.
    - Inference result persistence (``save_inference_results``).

    Subclasses must implement :meth:`evaluate_benchmark`.

    Args:
        model_name: Short label used for logging and output file paths.
        model_id: Model identifier (HuggingFace path or API model name).
        model_type: ``"pt"`` (pretrained) or ``"it"`` (instruction-tuned).
        thinking: Enable chain-of-thought / reasoning mode.
        system_prompt: Global system prompt override for all generative benchmarks.
        benchmark_system_prompts: Per-benchmark instruction strings.
        benchmark_answer_tags: Per-benchmark answer-tag prefix strings
            (e.g. ``"<answer>"``).
        results_dir: Root directory for writing per-model inference results.
    """

    def __init__(
        self,
        model_name: str,
        model_id: str,
        model_type: str,
        thinking: bool = False,
        system_prompt: Optional[str] = None,
        benchmark_system_prompts: Optional[dict] = None,
        benchmark_answer_tags: Optional[dict] = None,
        results_dir: str = "results",
    ):
        self.model_name = model_name
        self.model_id = model_id
        self.model_type = model_type
        self.thinking = thinking
        self.system_prompt = system_prompt
        self.benchmark_system_prompts = benchmark_system_prompts or {}
        self.benchmark_answer_tags = benchmark_answer_tags or {}
        self.results_dir = results_dir

    # ──────────────────────────────────────────────────────────────────────────
    # Abstract interface
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def evaluate_benchmark(
        self,
        benchmark_name: str,
        max_samples: Optional[int] = None,
        check_existing: bool = True,
        timestamp: Optional[str] = None,
    ) -> Optional[dict]:
        """Evaluate the model on a single benchmark. Must be implemented by subclasses."""

    # ──────────────────────────────────────────────────────────────────────────
    # Answer extraction (shared between VLLMEvaluator._async_gen and
    # CommercialEvaluator._process_batch_results)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_answer(
        response: str,
        answer_tag: Optional[str],
    ) -> tuple:
        """
        Parse a raw model response into ``(answer, thinking_trace, reflection)``.

        Strips ``<think>`` and ``<reflection>`` blocks first, then extracts the
        final answer either via an answer tag or the first non-empty line.

        Returns:
            A 3-tuple ``(answer: str, thinking_trace: str | None, reflection: str | None)``.
        """
        # ── Strip <think> block ──────────────────────────────────────────────
        thinking_trace = None
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking_trace = think_match.group(1).strip()
        response_for_answer = (
            re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
            if thinking_trace is not None
            else response
        )

        # ── Strip <reflection> block ─────────────────────────────────────────
        reflection = None
        ref_match = re.search(
            r"<reflection>(.*?)</reflection>",
            response_for_answer,
            re.DOTALL | re.IGNORECASE,
        )
        if ref_match:
            reflection = ref_match.group(1).strip()

        # ── Extract answer: tag → first line → whole response ────────────────
        if answer_tag and answer_tag.lower() in response_for_answer:
            _, _, after_tag = response_for_answer.partition(answer_tag.lower())
            closing = answer_tag.replace("<", "</").rstrip(">") + ">"
            if closing in after_tag:
                after_tag = after_tag.split(closing)[0]
            lines = after_tag.strip().splitlines()
            answer = lines[0].strip() if lines else ""
        elif "\n" in response_for_answer:
            lines = response_for_answer.splitlines()
            answer = lines[0].strip() if lines else response_for_answer
        else:
            answer = response_for_answer

        return answer, thinking_trace, reflection

    # ──────────────────────────────────────────────────────────────────────────
    # Metrics helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _calc_metrics(self, confidences: torch.Tensor) -> dict:
        """Overall MCQ metrics from an (N, K) log-prob tensor."""
        _, predicted = torch.max(confidences, 1)
        accuracy = (predicted == 0).float().mean().item()

        tp = (predicted == 0).float().sum().item()
        fn = (predicted != 0).float().sum().item()
        precision = tp / (tp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        softmaxed = F.softmax(confidences, dim=-1)
        path_confidence = softmaxed[:, 0].mean().item()
        num_options = confidences.shape[1]
        normalized_accuracy = (accuracy * num_options - 1) / (num_options - 1)

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "path_confidence": path_confidence,
            "normalized_accuracy": normalized_accuracy,
            "num_options": num_options,
        }

    @staticmethod
    def _group_by_category(results: list) -> dict:
        groups: dict = {}
        for r in results:
            cat = r.get("category") or "__all__"
            groups.setdefault(cat, []).append(r)
        return groups

    @staticmethod
    def _by_category_mcq(detailed: list) -> dict:
        out: dict = {}
        for cat, items in sorted(BaseEvaluator._group_by_category(detailed).items()):
            n = len(items)
            correct = sum(1 for r in items if r["is_correct"])
            acc = correct / n
            tp = correct
            fn = n - correct
            precision = tp / (tp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            path_conf_vals = []
            for r in items:
                lps = r.get("logprobs")
                if lps:
                    t = torch.tensor(lps, dtype=torch.float32)
                    path_conf_vals.append(F.softmax(t, dim=0)[0].item())
            path_confidence = (
                sum(path_conf_vals) / len(path_conf_vals) if path_conf_vals else 0.0
            )
            num_opts = len(items[0]["options"]) if items else 4
            norm_acc = (acc * num_opts - 1) / (num_opts - 1)

            out[cat] = {
                "num_samples": n,
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "path_confidence": round(path_confidence, 4),
                "normalized_accuracy": round(norm_acc, 4),
            }
        return out

    @staticmethod
    def _by_category_gen(detailed: list) -> dict:
        out: dict = {}
        for cat, items in sorted(BaseEvaluator._group_by_category(detailed).items()):
            n = len(items)
            out[cat] = {
                "num_samples": n,
                "exact_match": round(sum(r["exact_match"] for r in items) / n, 4),
                "contains_match": round(sum(r["contains_match"] for r in items) / n, 4),
                "prefix_match": round(sum(r["prefix_match"] for r in items) / n, 4),
            }
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # API key proxy support
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _maybe_wrap_key_for_proxy(api_key: str) -> str:
        """Wrap an API key using the API proxy if ``proxy_tools`` is installed."""
        try:
            from proxy_tools.api_key import get_api_key_for_proxy
            wrapped = get_api_key_for_proxy(api_key)
            print("  Using API key proxy")
            return wrapped
        except ImportError:
            return api_key

    # ──────────────────────────────────────────────────────────────────────────
    # Result persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save_inference_results(
        self,
        benchmark_name: str,
        detailed_results: list,
    ) -> str:
        """
        Write per-sample inference results to disk as JSONL.

        Returns:
            Path to the written file.
        """
        inference_dir = Path(self.results_dir) / self.model_name / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        inference_file = inference_dir / f"{benchmark_name}.jsonl"

        detailed_results.sort(key=lambda r: r.get("id", ""))
        with open(inference_file, "w", encoding="utf-8") as f:
            for result in detailed_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Saved inference results → {inference_file}")
        return str(inference_file)
