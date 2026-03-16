"""
AnthropicEvaluator — evaluates models via the Anthropic Message Batches API.

Requires the ``ANTHROPIC_API_KEY`` environment variable and ``anthropic >= 0.40``.
Batch processing offers a 50% cost discount with results delivered within 24 hours.
"""

import os
from typing import Optional

from .batch import BatchEvaluator


class AnthropicEvaluator(BatchEvaluator):
    """
    Evaluates Anthropic models using the Message Batches API.

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
        self.client = _anthropic.Anthropic(api_key=api_key)
        print(f"AnthropicEvaluator: Message Batches API → model={model_id}")

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
