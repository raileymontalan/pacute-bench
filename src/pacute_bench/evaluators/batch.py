"""
BatchEvaluator — abstract base for all commercial batch-API evaluators.

Implements the full submit-poll-collect lifecycle; subclasses only need to
implement two methods:

    _submit_batch(items, benchmark_name, effective_prompt) -> batch_id: str
    _try_collect_batch(batch_id) -> Optional[dict[sample_id, response_text]]

This eliminates the if/elif provider chains that existed in CommercialEvaluator
and makes adding a new provider a matter of creating one new file.
"""

import json
import time
from abc import abstractmethod
from pathlib import Path
from typing import Optional

from ..loaders import load_benchmark
from .base import BaseEvaluator, BENCHMARK_FORMATS


class BatchEvaluator(BaseEvaluator):
    """
    Abstract evaluator for providers that offer an asynchronous batch API.

    Only *generative* (``-gen``) benchmarks are supported — MCQ benchmarks
    require token log-probabilities which commercial APIs do not expose.

    Args:
        model_name: Short label for logging and output paths.
        model_id: API model identifier (e.g. ``"gpt-4o"``).
        thinking: Whether the model uses extended reasoning / chain-of-thought.
        system_prompt: Global system prompt override.
        benchmark_system_prompts: Per-benchmark system prompts from the evaluation config.
        benchmark_answer_tags: Per-benchmark answer-tag prefixes.
        results_dir: Root directory for inference JSONL output.
        poll_interval: Seconds between batch status polls (default: 60).
    """

    def __init__(
        self,
        model_name: str,
        model_id: str,
        thinking: bool = False,
        system_prompt: Optional[str] = None,
        benchmark_system_prompts: Optional[dict] = None,
        benchmark_answer_tags: Optional[dict] = None,
        results_dir: str = "results",
        poll_interval: int = 60,
    ):
        super().__init__(
            model_name=model_name,
            model_id=model_id,
            model_type="it",  # batch APIs are always instruction-tuned / chat
            thinking=thinking,
            system_prompt=system_prompt,
            benchmark_system_prompts=benchmark_system_prompts,
            benchmark_answer_tags=benchmark_answer_tags,
            results_dir=results_dir,
        )
        self.poll_interval = poll_interval

    # ──────────────────────────────────────────────────────────────────────────
    # Abstract interface — implement in each provider subclass
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def _submit_batch(
        self,
        items: list,
        benchmark_name: str,
        effective_prompt: Optional[str],
    ) -> str:
        """
        Submit a batch of requests and return a batch identifier immediately
        (non-blocking).

        Args:
            items: List of benchmark tuples ``(prefix, ground_truth, false_options, sample_id, ...)``.
            benchmark_name: Name of the benchmark being evaluated.
            effective_prompt: System/instruction prompt to prepend, or ``None``.

        Returns:
            An opaque string batch ID / job name used by :meth:`_try_collect_batch`.
        """

    @abstractmethod
    def _try_collect_batch(self, batch_id: str) -> Optional[dict]:
        """
        Non-blocking status check.

        Returns:
            A ``{sample_id: response_text}`` dict when the batch is complete,
            or ``None`` if it is still running.

        Raises:
            RuntimeError: If the batch ended in a failed / cancelled state.
        """

    # ──────────────────────────────────────────────────────────────────────────
    # Batch state persistence — survives PBS job restarts
    # ──────────────────────────────────────────────────────────────────────────

    def _batch_state_path(self, benchmark_name: str) -> Path:
        return (
            Path(self.results_dir) / self.model_name / ".batches" / f"{benchmark_name}.json"
        )

    def _save_batch_state(self, benchmark_name: str, state: dict) -> None:
        path = self._batch_state_path(benchmark_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"  [{benchmark_name}] Batch state saved → {path}")

    def _load_batch_state(self, benchmark_name: str) -> Optional[dict]:
        path = self._batch_state_path(benchmark_name)
        if not path.exists():
            return None
        with open(path) as f:
            state = json.load(f)
        print(f"  [{benchmark_name}] Resuming from saved batch state: {state.get('batch_id')}")
        return state

    def _delete_batch_state(self, benchmark_name: str) -> None:
        path = self._batch_state_path(benchmark_name)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # Public evaluation entry point — gen only
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_benchmark(
        self,
        benchmark_name: str,
        max_samples: Optional[int] = None,
        check_existing: bool = True,
        timestamp: Optional[str] = None,
    ) -> Optional[dict]:
        is_gen = BENCHMARK_FORMATS.get(benchmark_name, "mcq") == "gen"
        if not is_gen:
            print(f"\n  Skipping {benchmark_name} — MCQ not supported for batch evaluators.")
            return {"skipped": True, "reason": "mcq_not_supported"}

        print(f"\nEvaluating on {benchmark_name} (batch)...")

        try:
            benchmark_items = list(load_benchmark(benchmark_name))
        except Exception as exc:
            print(f"Error loading benchmark {benchmark_name}: {exc}")
            return None

        if max_samples:
            benchmark_items = benchmark_items[:max_samples]

        if not benchmark_items:
            print(f"No samples loaded for {benchmark_name}")
            return None

        print(f"Format: generative ({len(benchmark_items)} items)")

        if check_existing:
            inference_file = (
                Path(self.results_dir) / self.model_name / "inference" / f"{benchmark_name}.jsonl"
            )
            if inference_file.exists():
                print(f"  ⏭  Skipping — results exist at {inference_file}")
                print("       Use --overwrite to re-run.")
                return {"skipped": True, "inference_file": str(inference_file)}

        return self._evaluate_generative(benchmark_items, benchmark_name, "gen", timestamp)

    # ──────────────────────────────────────────────────────────────────────────
    # Parallel batch evaluation — submit all at once, poll together
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_benchmarks_parallel(
        self,
        benchmarks: list,
        max_samples: Optional[int] = None,
        check_existing: bool = True,
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Submit every gen benchmark as a batch simultaneously, then poll all
        concurrently until each one completes.  Returns
        ``{benchmark_name: results_dict}``.

        This is dramatically faster than calling ``evaluate_benchmark`` in a
        loop because batch processing time is dominated by the slowest single
        batch — not the sum of all batch times.
        """
        # ── Phase 1: submit all batches without waiting ──────────────────────
        pending: dict = {}  # benchmark_name -> {batch_id, items, answer_tag}

        for benchmark_name in benchmarks:
            if BENCHMARK_FORMATS.get(benchmark_name, "mcq") != "gen":
                print(f"  Skipping {benchmark_name} — MCQ not supported for batch evaluators.")
                continue

            try:
                items = list(load_benchmark(benchmark_name))
            except Exception as exc:
                print(f"  Error loading {benchmark_name}: {exc}")
                continue

            if max_samples:
                items = items[:max_samples]
            if not items:
                print(f"  No samples for {benchmark_name} — skipping.")
                continue

            if check_existing:
                inf_file = (
                    Path(self.results_dir) / self.model_name / "inference"
                    / f"{benchmark_name}.jsonl"
                )
                if inf_file.exists():
                    print(f"  Skipping {benchmark_name} — inference file already exists.")
                    continue

            bench_prompt = self.benchmark_system_prompts.get(benchmark_name)
            effective_prompt = (
                self.system_prompt if self.system_prompt is not None else bench_prompt
            )
            answer_tag = self.benchmark_answer_tags.get(benchmark_name)

            try:
                saved = self._load_batch_state(benchmark_name)
                if saved:
                    batch_id = saved["batch_id"]
                else:
                    batch_id = self._submit_batch(items, benchmark_name, effective_prompt)
                    self._save_batch_state(benchmark_name, {"batch_id": batch_id})
                pending[benchmark_name] = {
                    "batch_id": batch_id,
                    "items": items,
                    "answer_tag": answer_tag,
                }
            except Exception as exc:
                print(f"  Error submitting {benchmark_name}: {exc}")
                continue

        if not pending:
            return {}

        print(f"\n{'='*60}")
        print(f"All {len(pending)} batch(es) submitted. Polling every {self.poll_interval}s…")
        print(f"{'='*60}\n")

        # ── Phase 2: poll all concurrently until every batch finishes ────────
        results: dict = {}
        while pending:
            time.sleep(self.poll_interval)
            completed = []
            for bench, info in list(pending.items()):
                try:
                    results_by_id = self._try_collect_batch(info["batch_id"])
                except Exception as exc:
                    print(f"  Error checking {bench}: {exc}")
                    continue

                if results_by_id is not None:
                    results[bench] = self._process_batch_results(
                        info["items"], results_by_id,
                        info["answer_tag"], bench, "gen", timestamp,
                    )
                    self._delete_batch_state(bench)
                    completed.append(bench)
                    print(f"  ✓ {bench} done ({len(results_by_id)} results)")

            for bench in completed:
                del pending[bench]

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Single-benchmark dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_generative(self, items, benchmark_name, setting=None, timestamp=None):
        """Submit one batch and block until it completes."""
        bench_prompt = self.benchmark_system_prompts.get(benchmark_name)
        effective_prompt = self.system_prompt if self.system_prompt is not None else bench_prompt
        answer_tag = self.benchmark_answer_tags.get(benchmark_name)

        saved = self._load_batch_state(benchmark_name)
        if saved:
            batch_id = saved["batch_id"]
        else:
            batch_id = self._submit_batch(items, benchmark_name, effective_prompt)
            self._save_batch_state(benchmark_name, {"batch_id": batch_id})

        results_by_id = None
        while results_by_id is None:
            time.sleep(self.poll_interval)
            results_by_id = self._try_collect_batch(batch_id)

        self._delete_batch_state(benchmark_name)
        return self._process_batch_results(
            items, results_by_id, answer_tag, benchmark_name, setting, timestamp
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Result processing
    # ──────────────────────────────────────────────────────────────────────────

    def _process_batch_results(
        self, items, results_by_id: dict, answer_tag, benchmark_name, setting, timestamp
    ) -> dict:
        exact = contains = prefix_m = 0
        detailed = []

        for item_data in items:
            prefix, ground_truth, _, sample_id = item_data[:4]
            category = item_data[4] if len(item_data) > 4 else None

            response = results_by_id.get(str(sample_id), "")
            answer, thinking_trace, reflection = self._extract_answer(response, answer_tag)

            expected = self._normalize_label(str(ground_truth))
            answer   = self._normalize_label(answer)
            is_exact    = answer == expected
            is_contains = expected in answer
            is_prefix   = answer.startswith(expected)

            exact    += is_exact
            contains += is_contains
            prefix_m += is_prefix

            detailed.append({
                "id": sample_id,
                "category": category,
                "question": prefix,
                "ground_truth": ground_truth,
                "response": response,
                "thinking_trace": thinking_trace,
                "reflection": reflection,
                "answer": answer,
                "exact_match": is_exact,
                "contains_match": is_contains,
                "prefix_match": is_prefix,
            })

        n = len(detailed)
        if n == 0:
            return None

        return {
            "num_samples": n,
            "exact_match": exact / n,
            "contains_match": contains / n,
            "prefix_match": prefix_m / n,
            "format": "generative",
            "by_category": self._by_category_gen(detailed),
            "detailed_results": detailed,
            "setting": setting,
            "timestamp": timestamp,
        }
