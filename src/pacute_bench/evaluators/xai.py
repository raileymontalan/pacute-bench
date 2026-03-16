"""
XAIEvaluator — evaluates xAI Grok models via the xAI OpenAI-compatible API.

Requires the ``XAI_API_KEY`` environment variable.
Uses real-time async generation (no batch API available for xAI).
"""

import asyncio
import os
from typing import Optional

from openai import AsyncOpenAI

from .batch import BatchEvaluator


class XAIEvaluator(BatchEvaluator):
    """
    Evaluates xAI Grok models using the OpenAI-compatible xAI API.

    Uses real-time async generation (xAI does not offer a batch API).
    ``XAI_BASE_URL`` can be overridden via environment variable (default:
    ``"https://api.x.ai/v1"``); set it to route through the AISI proxy.

    Args:
        model_name: Short label for logging and output paths.
        model_id: API model identifier (e.g. ``"grok-3"``, ``"grok-3-mini"``).
        thinking: Whether to enable extended reasoning mode.
        system_prompt: Global system prompt override.
        benchmark_system_prompts: Per-benchmark system prompts.
        benchmark_answer_tags: Per-benchmark answer-tag prefixes.
        results_dir: Root directory for inference JSONL output.
        poll_interval: Unused (kept for interface compatibility with BatchEvaluator).
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

        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise EnvironmentError("XAI_API_KEY is not set")
        api_key = self._maybe_wrap_key_for_proxy(api_key)

        base_url = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        print(f"XAIEvaluator: async API → model={model_id} ({base_url})")

    # ──────────────────────────────────────────────────────────────────────────
    # Async generation (always used — no batch API for xAI)
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_benchmarks_parallel(self, benchmark_names, max_samples=None, check_existing=True, timestamp=None):
        """Run all benchmarks sequentially; each benchmark handles its own concurrency."""
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

    async def _run_async(
        self, items, benchmark_name: str, effective_prompt: Optional[str]
    ) -> dict:
        token_limit = 8192 if self.thinking else 256
        sem = asyncio.Semaphore(16)

        async def gen_one(sample_id: str, prefix: str):
            messages: list = []
            if effective_prompt:
                messages.append({"role": "system", "content": effective_prompt})
            messages.append({"role": "user", "content": str(prefix)})
            async with sem:
                response = await self.async_client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=token_limit,
                    temperature=0.0,
                )
            content = response.choices[0].message.content or ""
            return sample_id, content.strip().lower()

        tasks = [gen_one(str(item[3]), item[0]) for item in items]
        print(f"  Sending {len(tasks)} async requests (max 16 concurrent)…")
        results = await asyncio.gather(*tasks)
        return {sid: text for sid, text in results}

    # ──────────────────────────────────────────────────────────────────────────
    # BatchEvaluator abstract interface — not used for xAI (async only)
    # ──────────────────────────────────────────────────────────────────────────

    def _submit_batch(self, items, benchmark_name, effective_prompt):
        raise NotImplementedError("xAI does not support a batch API; use async generation.")

    def _try_collect_batch(self, batch_id):
        raise NotImplementedError("xAI does not support a batch API; use async generation.")
