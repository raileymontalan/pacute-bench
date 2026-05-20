"""
VLLMEvaluator — evaluates models against a running vLLM OpenAI-compatible server.

Supports both MCQ (log-probability scoring) and generative evaluation modes.
"""

import asyncio
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from ..loaders import load_benchmark
from .base import BaseEvaluator, BENCHMARK_FORMATS


class VLLMEvaluator(BaseEvaluator):
    """
    Evaluates language models on the pacute-bench suite via a running vLLM server.

    MCQ benchmarks are scored using log-probabilities (no generation required).
    Generative benchmarks use the chat-completions or completions endpoint.

    Args:
        model_name: Short label used for logging and output file paths.
        model_id: HuggingFace model id loaded by the vLLM server.
        model_type: ``"pt"`` (pretrained) or ``"it"`` (instruction-tuned).
        tokenizer_name: HuggingFace tokenizer for token counting during MCQ scoring.
        thinking: Enable chain-of-thought / reasoning mode.
            When True, ``max_new_tokens`` is set to 8192 and ``<think>``
            blocks are extracted from responses.
        reasoning_parser: Name of the vLLM server-side reasoning parser
            (e.g. ``"deepseek_r1"``), or ``None`` if none is configured.
            When ``None`` **and** ``thinking=True``, the Qwen3-style
            ``enable_thinking`` chat-template kwarg is passed instead.
            When set, ``<think>`` blocks are surfaced via
            ``message.reasoning_content`` rather than ``message.content``.
        vllm_url: Base URL of the vLLM server (default: ``"http://localhost:8000"``).
        vllm_api_key: API key for the vLLM server (default: ``"token-abc123"``).
        vllm_model_id: Override the model id used in API calls (useful when vLLM
            registers the model under a different name).
        system_prompt: Global system prompt override for all generative benchmarks.
        benchmark_system_prompts: Per-benchmark instruction strings loaded from
            ``configs/evaluation.yaml``.
        benchmark_answer_tags: Per-benchmark answer-tag prefix strings used to
            extract the answer from generated text (e.g. ``"<answer>"``).
        results_dir: Root directory for writing per-model inference results
            (default: ``"results"``).
    """

    def __init__(
        self,
        model_name: str,
        model_id: str,
        model_type: str,
        tokenizer_name: str,
        thinking: bool = False,
        reasoning_parser: Optional[str] = None,
        vllm_url: str = "http://localhost:8000",
        vllm_api_key: str = "token-abc123",
        vllm_model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        benchmark_system_prompts: Optional[dict] = None,
        benchmark_answer_tags: Optional[dict] = None,
        results_dir: str = "results",
    ):
        super().__init__(
            model_name=model_name,
            model_id=vllm_model_id or model_id,
            model_type=model_type,
            thinking=thinking,
            system_prompt=system_prompt,
            benchmark_system_prompts=benchmark_system_prompts,
            benchmark_answer_tags=benchmark_answer_tags,
            results_dir=results_dir,
        )
        self.reasoning_parser = reasoning_parser

        self.client = AsyncOpenAI(
            base_url=f"{vllm_url.rstrip('/')}/v1",
            api_key=vllm_api_key,
        )

        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if thinking:
            if reasoning_parser:
                thinking_label = f"thinking=ON via --reasoning-parser {reasoning_parser} (max_new_tokens=8192)"
            else:
                thinking_label = "thinking=ON via enable_thinking kwarg (max_new_tokens=8192)"
        else:
            thinking_label = "thinking=OFF (max_new_tokens=256)"
        registered = self.model_id
        if vllm_model_id and vllm_model_id != model_id:
            print(
                f"Connected to vLLM at {vllm_url} "
                f"(registered: {registered}, HF path: {model_id}, {thinking_label})"
            )
        else:
            print(f"Connected to vLLM at {vllm_url} (model: {registered}, {thinking_label})")

    # ──────────────────────────────────────────────────────────────────────────
    # Low-level async helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _compute_logprob(self, prefix: str, option: str) -> float:
        """Score log P(option | prefix) via the vLLM completions endpoint."""
        prefix = str(prefix)
        option = str(option)
        full_text = prefix + " " + option

        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=True)
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
        n_prefix = len(prefix_tokens)
        n_full = len(full_tokens)

        if n_full <= n_prefix:
            return -100.0

        response = await self.client.completions.create(
            model=self.model_id,
            prompt=full_text,
            max_tokens=1,
            echo=True,
            logprobs=1,
            temperature=0.0,
        )

        all_token_lps = response.choices[0].logprobs.token_logprobs
        option_token_lps = [lp for lp in all_token_lps[n_prefix:n_full] if lp is not None]
        return sum(option_token_lps) if option_token_lps else -100.0

    async def _score_options(self, prefix: str, ground_truth, false_options):
        """Score all MCQ options in parallel and return a log-prob tensor."""
        all_options = [ground_truth] + list(false_options)
        tasks = [self._compute_logprob(prefix, str(opt)) for opt in all_options]
        logprobs = await asyncio.gather(*tasks)
        return torch.tensor(list(logprobs))

    async def _generate(
        self,
        prefix: str,
        ground_truth,
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Generate text for one item via chat completions (IT) or completions (PT)."""
        prefix = str(prefix)
        ground_truth = str(ground_truth).strip().lower()

        if max_new_tokens is None:
            max_new_tokens = 8192 if self.thinking else 256

        effective_prompt = system_prompt if system_prompt is not None else self.system_prompt

        if self.model_type == "it":
            messages = []
            if effective_prompt:
                messages.append({"role": "system", "content": effective_prompt})
            messages.append({"role": "user", "content": prefix})

            # Qwen3 hybrid-thinking models use the enable_thinking chat-template kwarg.
            # Models with a server-side reasoning_parser (phi-4-reasoning, gpt-oss, DeepSeek-R1,
            # etc.) always emit <think> blocks; passing enable_thinking is unnecessary there.
            extra_body: dict = {}
            if self.thinking and self.reasoning_parser is None:
                extra_body["chat_template_kwargs"] = {"enable_thinking": True}

            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
                **(dict(extra_body=extra_body) if extra_body else {}),
            )
            msg = response.choices[0].message
            content = msg.content
            # When a reasoning_parser is active on the vLLM server, <think> blocks are
            # moved into message.reasoning_content; message.content is the clean answer.
            thinking_content: Optional[str] = getattr(msg, "reasoning_content", None)
            generated = content.strip().lower() if content is not None else ""
        else:
            prompt = (effective_prompt + "\n\n" + prefix) if effective_prompt else prefix
            response = await self.client.completions.create(
                model=self.model_id,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
            text = response.choices[0].text
            generated = text.strip().lower() if text is not None else ""
            thinking_content = None

        return {
            "generated": generated,
            "thinking_content": thinking_content,
            "ground_truth": ground_truth,
            "exact_match": generated == ground_truth,
            "contains_match": ground_truth in generated,
            "prefix_match": generated.startswith(ground_truth),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Public evaluation entry point
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_benchmark(
        self,
        benchmark_name: str,
        max_samples: Optional[int] = None,
        check_existing: bool = True,
        timestamp: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Evaluate the model on a single benchmark.

        Args:
            benchmark_name: Registered benchmark name (e.g. ``"pacute-composition-mcq"``).
            max_samples: Cap on number of items (``None`` = all).
            check_existing: Skip and return ``{"skipped": True}`` when inference
                results already exist on disk.
            timestamp: Timestamp string for output filenames.

        Returns:
            Result dict, ``{"skipped": True, ...}``, or ``None`` on error.
        """
        print(f"\nEvaluating on {benchmark_name}...")

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

        is_gen = BENCHMARK_FORMATS.get(benchmark_name, "mcq") == "gen"
        print(f"Format: {'generative' if is_gen else 'MCQ'} ({len(benchmark_items)} items)")

        if check_existing:
            inference_file = (
                Path(self.results_dir) / self.model_name / "inference" / f"{benchmark_name}.jsonl"
            )
            if inference_file.exists():
                print(f"  ⏭  Skipping — results exist at {inference_file}")
                print("       Use --overwrite to re-run.")
                return {"skipped": True, "inference_file": str(inference_file)}

        setting = "gen" if is_gen else "mcq"
        if is_gen:
            return self._evaluate_generative(benchmark_items, benchmark_name, setting, timestamp)
        else:
            return self._evaluate_mcq(benchmark_items, benchmark_name, setting, timestamp)

    # ──────────────────────────────────────────────────────────────────────────
    # MCQ evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_mcq(self, items, benchmark_name, setting=None, timestamp=None):
        return asyncio.run(self._async_mcq(items, benchmark_name, setting, timestamp))

    async def _async_mcq(self, items, benchmark_name, setting=None, timestamp=None):
        sem = asyncio.Semaphore(16)
        pbar = tqdm(total=len(items), desc=benchmark_name)

        async def score_item(item):
            prefix, ground_truth, false_options, sample_id = item[:4]
            category = item[4] if len(item) > 4 else None
            async with sem:
                logprobs = await self._score_options(prefix, ground_truth, false_options)
            pbar.update(1)
            predicted_idx = torch.argmax(logprobs).item()
            all_options = [ground_truth] + list(false_options)
            return {
                "logprobs": logprobs,
                "detail": {
                    "id": sample_id,
                    "category": category,
                    "question": prefix,
                    "ground_truth": ground_truth,
                    "options": all_options,
                    "predicted_idx": predicted_idx,
                    "predicted_answer": (
                        all_options[predicted_idx] if predicted_idx < len(all_options) else None
                    ),
                    "is_correct": predicted_idx == 0,
                    "logprobs": logprobs.tolist(),
                },
            }

        scored = await asyncio.gather(*[score_item(item) for item in items])
        pbar.close()

        confidences = [s["logprobs"] for s in scored]
        detailed = [s["detail"] for s in scored]
        n = len(scored)

        max_len = max(len(c) for c in confidences)
        padded = [F.pad(c, (0, max_len - len(c)), value=-1e10) for c in confidences]
        conf_tensor = torch.stack(padded)

        results = self._calc_metrics(conf_tensor)
        results.update({
            "num_samples": n,
            "format": "mcq",
            "by_category": self._by_category_mcq(detailed),
            "detailed_results": detailed,
            "setting": setting,
            "timestamp": timestamp,
        })
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Generative evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_generative(self, items, benchmark_name, setting=None, timestamp=None):
        return asyncio.run(self._async_gen(items, benchmark_name, setting, timestamp))

    async def _async_gen(self, items, benchmark_name, setting=None, timestamp=None):
        bench_prompt = self.benchmark_system_prompts.get(benchmark_name)
        effective_prompt = (
            self.system_prompt if self.system_prompt is not None else bench_prompt
        )
        if effective_prompt is not None:
            src = "CLI override" if self.system_prompt is not None else "evaluation config"
            print(f"  Using instruction for {benchmark_name} (source: {src})")

        answer_tag = self.benchmark_answer_tags.get(benchmark_name)

        sem = asyncio.Semaphore(64)

        async def gen_one(prefix, ground_truth):
            async with sem:
                return await self._generate(str(prefix), ground_truth, system_prompt=effective_prompt)

        tasks = [gen_one(item[0], item[1]) for item in items]
        print(f"  Sending {len(tasks)} generation requests (max 64 concurrent)...")
        api_responses = await asyncio.gather(*tasks)

        exact = contains = prefix_m = 0
        detailed = []

        for item_data, api_resp in zip(items, api_responses):
            prefix, ground_truth, _, sample_id = item_data[:4]
            category = item_data[4] if len(item_data) > 4 else None

            response = api_resp["generated"]
            answer, thinking_trace, reflection = self._extract_answer(response, answer_tag)
            # When a server-side reasoning_parser is active, <think> content is in
            # thinking_content (message.reasoning_content) rather than in the response text.
            if api_resp.get("thinking_content") is not None:
                thinking_trace = api_resp["thinking_content"]

            expected = self._normalize_label(str(ground_truth))
            answer   = self._normalize_label(answer)
            is_exact = answer == expected
            is_contains = expected in answer
            is_prefix = answer.startswith(expected)

            exact += is_exact
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
