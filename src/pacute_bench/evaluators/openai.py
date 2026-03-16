"""
OpenAIEvaluator — evaluates models via the OpenAI batch API.

Requires the ``OPENAI_API_KEY`` environment variable and ``openai >= 1.30``.
Batch processing offers a 50% cost discount with results delivered within 24 hours.
"""

import json
import os
import tempfile
from typing import Optional

from .batch import BatchEvaluator


class OpenAIEvaluator(BatchEvaluator):
    """
    Evaluates OpenAI models using the Batch API.

    Args:
        model_name: Short label for logging and output paths.
        model_id: API model identifier (e.g. ``"gpt-4o"``, ``"gpt-5"``).
        thinking: Whether the model uses extended reasoning (o-series models).
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

        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        print(f"OpenAIEvaluator: batch API → model={model_id}")

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
        max_tokens = 8192 if self.thinking else 256
        batch_requests = []
        for item in items:
            prefix, _gt, _, sample_id = item[:4]
            messages: list = []
            if effective_prompt:
                messages.append({"role": "system", "content": effective_prompt})
            messages.append({"role": "user", "content": str(prefix)})
            batch_requests.append({
                "custom_id": str(sample_id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_id,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                    "temperature": 0.0,
                },
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
