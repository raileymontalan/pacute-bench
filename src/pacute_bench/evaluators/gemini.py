"""
GeminiEvaluator — evaluates models via the Google Gemini Batch API.

Requires the ``GEMINI_API_KEY`` environment variable and ``google-genai >= 0.8``.
"""

import os
from typing import Optional

from .batch import BatchEvaluator


class GeminiEvaluator(BatchEvaluator):
    """
    Evaluates Google Gemini models using the Gemini Batch API.

    Args:
        model_name: Short label for logging and output paths.
        model_id: API model identifier (e.g. ``"gemini-2.5-pro-preview-06-05"``).
        thinking: Whether to enable extended thinking (where supported).
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

        from google import genai as _genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set")
        self.client = _genai.Client(api_key=api_key)
        print(f"GeminiEvaluator: Batch API → model={model_id}")

    # ──────────────────────────────────────────────────────────────────────────
    # BatchEvaluator interface
    # ──────────────────────────────────────────────────────────────────────────

    def _submit_batch(
        self,
        items: list,
        benchmark_name: str,
        effective_prompt: Optional[str],
    ) -> str:
        """Create a Gemini batch job. Returns the job name immediately."""
        from google.genai import types as _gtypes

        max_tokens = 8192 if self.thinking else 256
        requests = []
        for item in items:
            prefix, _gt, _, sample_id = item[:4]
            generate_request: dict = {
                "contents": [{"role": "user", "parts": [{"text": str(prefix)}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.0,
                },
            }
            if effective_prompt:
                generate_request["systemInstruction"] = {
                    "parts": [{"text": effective_prompt}]
                }
            requests.append({
                "custom_id": str(sample_id),
                "generateContentRequest": generate_request,
            })

        print(f"  [{benchmark_name}] Submitting {len(requests)} requests to Gemini…")
        batch = self.client.batches.create(
            model=self.model_id,
            src=requests,
            config=_gtypes.CreateBatchJobConfig(
                display_name=f"pacute-{benchmark_name[:40]}",
            ),
        )
        print(f"  [{benchmark_name}] Batch submitted: {batch.name}")
        return batch.name

    def _try_collect_batch(self, batch_id: str) -> Optional[dict]:
        """
        Non-blocking check. Returns ``results_by_id`` when the job reaches
        ``JOB_STATE_SUCCEEDED``, ``None`` if still running, or raises
        ``RuntimeError`` for failed/cancelled jobs.
        """
        batch = self.client.batches.get(name=batch_id)
        state_str = batch.state.name if hasattr(batch.state, "name") else str(batch.state)
        counts = getattr(batch, "request_counts", None)
        counts_str = (
            f"  succeeded={counts.succeeded}  failed={counts.failed}" if counts else ""
        )
        print(f"  [{batch_id}] state={state_str}{counts_str}")

        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
        if state_str not in terminal:
            return None

        if state_str != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Gemini batch {batch_id} ended with state: {state_str}")

        results_by_id: dict = {}
        for response in batch.responses:
            cid = response.custom_id
            try:
                text = response.response.candidates[0].content.parts[0].text or ""
                results_by_id[cid] = text.strip().lower()
            except Exception as exc:
                print(f"  Warning: request {cid} failed: {exc}")
                results_by_id[cid] = ""

        return results_by_id
