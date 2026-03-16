"""
OpenAIEvaluator — evaluates models via the OpenAI batch API.

Requires the ``OPENAI_API_KEY`` environment variable and ``openai >= 1.30``.
Batch processing offers a 50% cost discount with results delivered within 24 hours.

When ``OPENAI_BASE_URL`` is set (e.g. for the AISI API key proxy), the batch
results endpoint is typically unavailable; the evaluator automatically falls
back to real-time async generation instead.
"""

import asyncio
import json
import os
import tempfile
from typing import Optional

from openai import AsyncOpenAI

from .batch import BatchEvaluator


class OpenAIEvaluator(BatchEvaluator):
    """
    Evaluates OpenAI models using the Batch API.

    When ``OPENAI_BASE_URL`` is set (e.g. the AISI API key proxy), the evaluator
    automatically switches to real-time async generation because the batch
    results endpoint is not exposed by the proxy.

    Reasoning models (o1, o3, o4, gpt-5 families) are handled automatically:
    system prompts are prepended as an initial user message, and
    ``max_completion_tokens`` is used without a ``temperature`` setting.

    Args:
        model_name: Short label for logging and output paths.
        model_id: API model identifier (e.g. ``"gpt-4o"``, ``"gpt-5"``).
        thinking: Whether the model uses extended reasoning (o-series models).
        force_async: Always use real-time async generation instead of the batch
            API (used for models where batch turnaround is impractical, e.g. gpt-5).
        system_prompt: Global system prompt override.
        benchmark_system_prompts: Per-benchmark system prompts.
        benchmark_answer_tags: Per-benchmark answer-tag prefixes.
        results_dir: Root directory for inference JSONL output.
        poll_interval: Seconds between batch status polls (default: 60).
    """

    # Reasoning model prefixes — require max_completion_tokens and no system role.
    _REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    def __init__(
        self,
        model_name: str,
        model_id: str,
        thinking: bool = False,
        force_async: bool = False,
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

        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        api_key = self._maybe_wrap_key_for_proxy(api_key)

        base_url = os.environ.get("OPENAI_BASE_URL")
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        # _use_async=True when: routing through a proxy (batch results endpoint
        # unavailable) OR explicitly forced (e.g. gpt-5 family where batch
        # turnaround is too slow).
        self._use_async = bool(base_url) or force_async
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

        if force_async and not base_url:
            mode_label = "async (forced)"
        elif self._use_async:
            mode_label = "async (proxy)"
        else:
            mode_label = "batch"
        proxy_note = f" via {base_url}" if base_url else ""
        print(f"OpenAIEvaluator: {mode_label} API → model={model_id}{proxy_note}")

    def _is_reasoning_model(self) -> bool:
        return any(self.model_id.startswith(p) for p in self._REASONING_PREFIXES)

    # ──────────────────────────────────────────────────────────────────────────
    # Async path — used automatically when OPENAI_BASE_URL is set
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_benchmarks_parallel(self, benchmark_names, max_samples=None, check_existing=True, timestamp=None):
        if not self._use_async:
            return super().evaluate_benchmarks_parallel(
                benchmark_names, max_samples=max_samples,
                check_existing=check_existing, timestamp=timestamp,
            )
        # Async mode: each benchmark is already handled concurrently internally;
        # just run evaluate_benchmark sequentially across benchmarks.
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
        is_reasoning = self._is_reasoning_model()
        token_limit = 8192 if (self.thinking or is_reasoning) else 256
        sem = asyncio.Semaphore(16)

        async def gen_one(sample_id: str, prefix: str):
            messages: list = []
            if effective_prompt:
                if is_reasoning:
                    # Reasoning models don't support system role messages.
                    messages.append({"role": "user", "content": effective_prompt})
                else:
                    messages.append({"role": "system", "content": effective_prompt})
            messages.append({"role": "user", "content": str(prefix)})
            kwargs: dict = {"model": self.model_id, "messages": messages,
                            "max_completion_tokens": token_limit}
            if not is_reasoning:
                kwargs["temperature"] = 0.0
            async with sem:
                response = await self.async_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            return sample_id, content.strip().lower()

        tasks = [gen_one(str(item[3]), item[0]) for item in items]
        print(f"  Sending {len(tasks)} async requests (max 16 concurrent)…")
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
        """Upload requests and create a batch. Returns the batch_id immediately."""
        is_reasoning = self._is_reasoning_model()
        token_limit = 8192 if (self.thinking or is_reasoning) else 256
        batch_requests = []
        for item in items:
            prefix, _gt, _, sample_id = item[:4]
            messages: list = []
            if effective_prompt:
                if is_reasoning:
                    # Reasoning models don't support system role messages.
                    messages.append({"role": "user", "content": effective_prompt})
                else:
                    messages.append({"role": "system", "content": effective_prompt})
            messages.append({"role": "user", "content": str(prefix)})
            body: dict = {
                "model": self.model_id,
                "messages": messages,
                "max_completion_tokens": token_limit,
            }
            if not is_reasoning:
                body["temperature"] = 0.0
            batch_requests.append({
                "custom_id": str(sample_id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix=f"pacute_{benchmark_name}_"
        ) as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")
            tmp_path = f.name

        print(f"  [{benchmark_name}] Uploading {len(batch_requests)} requests…")
        with open(tmp_path, "rb") as fh:
            file_obj = self.client.files.create(file=fh, purpose="batch")

        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"benchmark": benchmark_name, "model": self.model_name},
        )
        print(f"  [{benchmark_name}] Batch submitted: {batch.id}")
        return batch.id

    def _try_collect_batch(self, batch_id: str) -> Optional[dict]:
        """
        Non-blocking check. Returns ``results_by_id`` when done, ``None`` if still
        running. Raises ``RuntimeError`` for failed/expired/cancelled batches.
        """
        batch = self.client.batches.retrieve(batch_id)
        terminal = {"completed", "failed", "expired", "cancelled"}

        rc = batch.request_counts
        print(
            f"  [{batch_id}] status={batch.status}  "
            f"completed={rc.completed}/{rc.total}  failed={rc.failed}"
        )

        if batch.status not in terminal:
            return None

        if batch.status != "completed":
            raise RuntimeError(f"OpenAI batch {batch_id} ended with status: {batch.status}")

        output_text = self.client.files.content(batch.output_file_id).text
        results_by_id: dict = {}
        for line in output_text.strip().splitlines():
            obj = json.loads(line)
            cid = obj["custom_id"]
            if obj["response"]["status_code"] == 200:
                content = obj["response"]["body"]["choices"][0]["message"]["content"] or ""
                results_by_id[cid] = content.strip().lower()
            else:
                print(f"  Warning: request {cid} failed: {obj['response']}")
                results_by_id[cid] = ""

        # Best-effort cleanup of the uploaded input file.
        # The batch object always carries input_file_id.
        try:
            if batch.input_file_id:
                self.client.files.delete(batch.input_file_id)
        except Exception:
            pass

        return results_by_id
