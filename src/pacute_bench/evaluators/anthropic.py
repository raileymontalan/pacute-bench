"""
AnthropicEvaluator — evaluates models via the Anthropic Message Batches API.

Requires the ``ANTHROPIC_API_KEY`` environment variable and ``anthropic >= 0.40``.
Batch processing offers a 50% cost discount with results delivered within 24 hours.

When ``ANTHROPIC_BASE_URL`` is set (e.g. for the API key proxy), the batch
results endpoint is unavailable; the evaluator automatically falls back to
real-time async generation instead.
"""

import asyncio
import os
from typing import Optional

from .batch import BatchEvaluator


class AnthropicEvaluator(BatchEvaluator):
    """
    Evaluates Anthropic models using the Message Batches API.

    When ``ANTHROPIC_BASE_URL`` is set (e.g. the API key proxy), the evaluator
    automatically switches to real-time async generation because the batch
    results endpoint is not exposed by the proxy.

    Args:
        model_name: Short label for logging and output paths.
        model_id: API model identifier (e.g. ``"claude-sonnet-4-6"``).
        thinking: Whether the model uses extended thinking (claude-3-7+ models).
        system_prompt: Global system prompt override.
        benchmark_system_prompts: Per-benchmark system prompts.
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
            thinking=thinking,
            system_prompt=system_prompt,
            benchmark_system_prompts=benchmark_system_prompts,
            benchmark_answer_tags=benchmark_answer_tags,
            results_dir=results_dir,
            poll_interval=poll_interval,
        )

        import anthropic as _anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set")
        api_key = self._maybe_wrap_key_for_proxy(api_key)

        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = _anthropic.Anthropic(**client_kwargs)
        self.async_anthropic = _anthropic.AsyncAnthropic(**client_kwargs)
        # Use async generation when going through a proxy (batch results
        # endpoint is not supported by the API proxy).
        self._use_async = bool(base_url)

        mode_label = "async (proxy)" if self._use_async else "batch"
        proxy_note = f" via {base_url}" if base_url else ""
        print(f"AnthropicEvaluator: {mode_label} API → model={model_id}{proxy_note}")

    # ──────────────────────────────────────────────────────────────────────────
    # Async path — used automatically when ANTHROPIC_BASE_URL is set
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_benchmarks_parallel(self, benchmark_names, max_samples=None, check_existing=True, timestamp=None):
        if not self._use_async:
            return super().evaluate_benchmarks_parallel(
                benchmark_names, max_samples=max_samples,
                check_existing=check_existing, timestamp=timestamp,
            )
        results = {}
        for bench in benchmark_names:
            r = self.evaluate_benchmark(
                bench, max_samples=max_samples,
                check_existing=check_existing, timestamp=timestamp,
            )
            if r and not r.get("skipped"):
                results[bench] = r
        return results

    def _evaluate_generative(self, items, benchmark_name, setting=None, timestamp=None):
        if self._use_async:
            bench_prompt = self.benchmark_system_prompts.get(benchmark_name)
            effective_prompt = (
                self.system_prompt if self.system_prompt is not None else bench_prompt
            )
            answer_tag = self.benchmark_answer_tags.get(benchmark_name)
            results_by_id = asyncio.run(
                self._run_async(items, benchmark_name, effective_prompt)
            )
            return self._process_batch_results(
                items, results_by_id, answer_tag, benchmark_name, setting, timestamp
            )
        return super()._evaluate_generative(items, benchmark_name, setting, timestamp)

    async def _run_async(
        self, items, benchmark_name: str, effective_prompt: Optional[str]
    ) -> dict:
        max_tokens = 8192 if self.thinking else 256
        max_concurrent = int(os.environ.get("PACUTE_BENCH_MAX_CONCURRENT", 16))
        sem = asyncio.Semaphore(max_concurrent)

        async def gen_one(sample_id: str, prefix: str):
            kwargs: dict = {
                "model": self.model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": str(prefix)}],
            }
            if effective_prompt:
                kwargs["system"] = effective_prompt
            async with sem:
                response = await self.async_anthropic.messages.create(**kwargs)
            text = response.content[0].text or ""
            return sample_id, text.strip().lower()

        tasks = [gen_one(str(item[3]), item[0]) for item in items]
        print(f"  Sending {len(tasks)} async requests (max {max_concurrent} concurrent)…")
        results = await asyncio.gather(*tasks)
        return {sid: text for sid, text in results}

    # ──────────────────────────────────────────────────────────────────────────
    # BatchEvaluator interface
    # ──────────────────────────────────────────────────────────────────────────

    def _submit_batch(
        self,
        items: list,
        benchmark_name: str,
        effective_prompt: Optional[str],
    ) -> str:
        """Create an Anthropic message batch. Returns the batch_id immediately."""
        max_tokens = 8192 if self.thinking else 256
        requests = []
        for item in items:
            prefix, _gt, _, sample_id = item[:4]
            params: dict = {
                "model": self.model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": str(prefix)}],
            }
            if effective_prompt:
                params["system"] = effective_prompt
            requests.append({"custom_id": str(sample_id), "params": params})

        print(f"  [{benchmark_name}] Submitting {len(requests)} requests…")
        batch = self.client.messages.batches.create(requests=requests)
        print(f"  [{benchmark_name}] Batch submitted: {batch.id}")
        return batch.id

    def _try_collect_batch(self, batch_id: str) -> Optional[dict]:
        """
        Non-blocking check. Returns ``results_by_id`` when ``processing_status
        == "ended"``, ``None`` otherwise.
        """
        batch = self.client.messages.batches.retrieve(batch_id)
        rc = batch.request_counts
        print(
            f"  [{batch_id}] status={batch.processing_status}  "
            f"succeeded={rc.succeeded}  errored={rc.errored}  "
            f"processing={rc.processing}"
        )

        if batch.processing_status != "ended":
            return None

        results_by_id: dict = {}
        for result in self.client.messages.batches.results(batch_id):
            cid = result.custom_id
            if result.result.type == "succeeded":
                text = result.result.message.content[0].text or ""
                results_by_id[cid] = text.strip().lower()
            else:
                print(f"  Warning: request {cid} failed: {result.result}")
                results_by_id[cid] = ""

        return results_by_id
