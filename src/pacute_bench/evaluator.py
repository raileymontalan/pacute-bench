"""
VLLMEvaluator — runs benchmark evaluations against a running vLLM server.

Supports both MCQ (log-probability scoring) and generative evaluation modes.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from .loaders import load_benchmark, BENCHMARK_REGISTRY

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

    "pacute-affixation":         "mcq",
    "pacute-affixation-mcq":     "mcq",
    "pacute-affixation-gen":     "gen",
    "pacute-composition":        "mcq",
    "pacute-composition-mcq":    "mcq",
    "pacute-composition-gen":    "gen",
    "pacute-manipulation":       "mcq",
    "pacute-manipulation-mcq":   "mcq",
    "pacute-manipulation-gen":   "gen",
    "pacute-syllabification":    "mcq",
    "pacute-syllabification-mcq": "mcq",
    "pacute-syllabification-gen": "gen",
}


class VLLMEvaluator:
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
            When True, ``max_new_tokens`` is set to 8192, the vLLM
            ``enable_thinking`` chat-template kwarg is passed as ``True``,
            and ``<think>`` blocks are extracted from responses.
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
        vllm_url: str = "http://localhost:8000",
        vllm_api_key: str = "token-abc123",
        vllm_model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        benchmark_system_prompts: Optional[dict] = None,
        benchmark_answer_tags: Optional[dict] = None,
        results_dir: str = "results",
    ):
        self.model_name = model_name
        self.model_id = vllm_model_id or model_id
        self.model_type = model_type
        self.thinking = thinking
        self.system_prompt = system_prompt
        self.benchmark_system_prompts = benchmark_system_prompts or {}
        self.benchmark_answer_tags = benchmark_answer_tags or {}
        self.results_dir = results_dir

        self.client = AsyncOpenAI(
            base_url=f"{vllm_url.rstrip('/')}/v1",
            api_key=vllm_api_key,
        )

        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        thinking_label = (
            "thinking=ON (max_new_tokens=8192)"
            if thinking
            else "thinking=OFF (max_new_tokens=256)"
        )
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
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
            )
            content = response.choices[0].message.content
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

        return {
            "generated": generated,
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
            benchmark_name: Registered benchmark name (e.g. ``"pacute-affixation-mcq"``).
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

            # Extract <think> block (chain-of-thought / reasoning models).
            thinking_trace = None
            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
            if think_match:
                thinking_trace = think_match.group(1).strip()
            response_for_answer = (
                re.sub(
                    r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE
                ).strip()
                if thinking_trace is not None
                else response
            )

            # Extract <reflection> block.
            reflection = None
            ref_match = re.search(
                r"<reflection>(.*?)</reflection>",
                response_for_answer,
                re.DOTALL | re.IGNORECASE,
            )
            if ref_match:
                reflection = ref_match.group(1).strip()

            # Extract the answer: strip answer-tag, then take first line.
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

            expected = str(ground_truth).strip().lower()
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
        for cat, items in sorted(VLLMEvaluator._group_by_category(detailed).items()):
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
        for cat, items in sorted(VLLMEvaluator._group_by_category(detailed).items()):
            n = len(items)
            out[cat] = {
                "num_samples": n,
                "exact_match": round(sum(r["exact_match"] for r in items) / n, 4),
                "contains_match": round(sum(r["contains_match"] for r in items) / n, 4),
                "prefix_match": round(sum(r["prefix_match"] for r in items) / n, 4),
            }
        return out

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


# ---------------------------------------------------------------------------
# CommercialEvaluator — OpenAI and Anthropic via their batch APIs (50% off)
# ---------------------------------------------------------------------------

class CommercialEvaluator(VLLMEvaluator):
    """
    Evaluates commercial models (OpenAI, Anthropic) using their batch APIs.

    Both providers offer a 50 % cost discount via asynchronous batch processing
    with results delivered within 24 hours.  Only *generative* (``-gen``)
    benchmarks are supported — MCQ benchmarks require token log-probabilities
    which commercial APIs do not expose.

    Args:
        model_name: Short label for logging and output paths.
        model_id: API model identifier (e.g. ``"gpt-4o"`` or ``"claude-opus-4-6"``).
        provider: ``"openai"`` or ``"anthropic"``.
        thinking: Whether the model uses extended reasoning / chain-of-thought.
        system_prompt: Global system prompt override.
        benchmark_system_prompts: Per-benchmark system prompts from evaluation config.
        benchmark_answer_tags: Per-benchmark answer-tag prefixes.
        results_dir: Root directory for inference JSONL output.
        poll_interval: Seconds between batch status polls (default: 60).
    """

    def __init__(
        self,
        model_name: str,
        model_id: str,
        provider: str,
        thinking: bool = False,
        system_prompt: Optional[str] = None,
        benchmark_system_prompts: Optional[dict] = None,
        benchmark_answer_tags: Optional[dict] = None,
        results_dir: str = "results",
        poll_interval: int = 60,
    ):
        # Bypass VLLMEvaluator.__init__ entirely — no vLLM server, no tokenizer.
        self.model_name = model_name
        self.model_id = model_id
        self.model_type = "it"   # commercial APIs are always chat / instruction models
        self.thinking = thinking
        self.system_prompt = system_prompt
        self.benchmark_system_prompts = benchmark_system_prompts or {}
        self.benchmark_answer_tags = benchmark_answer_tags or {}
        self.results_dir = results_dir
        self.poll_interval = poll_interval
        self.provider = provider.lower()

        if self.provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY is not set")
            self.client = OpenAI(api_key=api_key)
            print(f"CommercialEvaluator: OpenAI batch API → model={model_id}")

        elif self.provider == "anthropic":
            import anthropic as _anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY is not set")
            self.client = _anthropic.Anthropic(api_key=api_key)
            print(f"CommercialEvaluator: Anthropic batch API → model={model_id}")

        else:
            raise ValueError(f"Unknown provider '{provider}'. Expected 'openai' or 'anthropic'.")

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
            print(f"\n  Skipping {benchmark_name} — MCQ not supported for commercial models.")
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
    # Batch state persistence (resume after PBS job restart)
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
    # Parallel batch evaluation (submit all benchmarks at once, poll together)
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
        import time

        # ── Phase 1: submit all batches without waiting ──────────────────────
        pending: dict = {}  # benchmark_name -> submission metadata

        for benchmark_name in benchmarks:
            if BENCHMARK_FORMATS.get(benchmark_name, "mcq") != "gen":
                print(f"  Skipping {benchmark_name} — MCQ not supported for commercial models.")
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
                    # Resume: reuse existing batch, skip re-submission.
                    pending[benchmark_name] = {"items": items, "answer_tag": answer_tag, **saved}
                elif self.provider == "openai":
                    info = self._submit_openai_batch(items, benchmark_name, effective_prompt)
                    state = {"batch_id": info["batch_id"], "file_id": info["file_id"]}
                    self._save_batch_state(benchmark_name, state)
                    pending[benchmark_name] = {"items": items, "answer_tag": answer_tag, **state}
                else:
                    batch_id = self._submit_anthropic_batch(items, benchmark_name, effective_prompt)
                    state = {"batch_id": batch_id}
                    self._save_batch_state(benchmark_name, state)
                    pending[benchmark_name] = {"items": items, "answer_tag": answer_tag, **state}
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
                    if self.provider == "openai":
                        batch = self.client.batches.retrieve(info["batch_id"])
                        rc = batch.request_counts
                        print(
                            f"  [{bench}] {info['batch_id']}  status={batch.status}  "
                            f"completed={rc.completed}/{rc.total}  failed={rc.failed}"
                        )
                        results_by_id = self._try_collect_openai_batch(
                            info["batch_id"], info["file_id"]
                        )
                    else:
                        batch = self.client.messages.batches.retrieve(info["batch_id"])
                        rc = batch.request_counts
                        print(
                            f"  [{bench}] {info['batch_id']}  "
                            f"status={batch.processing_status}  "
                            f"succeeded={rc.succeeded}  errored={rc.errored}  "
                            f"processing={rc.processing}"
                        )
                        results_by_id = self._try_collect_anthropic_batch(info["batch_id"])
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
    # Single-benchmark dispatch (used by evaluate_benchmark)
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_generative(self, items, benchmark_name, setting=None, timestamp=None):
        """Submit one batch and block until it completes."""
        import time
        bench_prompt = self.benchmark_system_prompts.get(benchmark_name)
        effective_prompt = self.system_prompt if self.system_prompt is not None else bench_prompt
        answer_tag = self.benchmark_answer_tags.get(benchmark_name)

        if self.provider == "openai":
            saved = self._load_batch_state(benchmark_name)
            if saved:
                info = saved
            else:
                info = self._submit_openai_batch(items, benchmark_name, effective_prompt)
                self._save_batch_state(benchmark_name, {"batch_id": info["batch_id"], "file_id": info["file_id"]})
            results_by_id = None
            while results_by_id is None:
                time.sleep(self.poll_interval)
                batch = self.client.batches.retrieve(info["batch_id"])
                rc = batch.request_counts
                print(
                    f"  [{info['batch_id']}] status={batch.status}  "
                    f"completed={rc.completed}/{rc.total}  failed={rc.failed}"
                )
                results_by_id = self._try_collect_openai_batch(
                    info["batch_id"], info["file_id"]
                )
        else:
            saved = self._load_batch_state(benchmark_name)
            if saved:
                batch_id = saved["batch_id"]
            else:
                batch_id = self._submit_anthropic_batch(items, benchmark_name, effective_prompt)
                self._save_batch_state(benchmark_name, {"batch_id": batch_id})
            results_by_id = None
            while results_by_id is None:
                time.sleep(self.poll_interval)
                batch = self.client.messages.batches.retrieve(batch_id)
                rc = batch.request_counts
                print(
                    f"  [{batch_id}] status={batch.processing_status}  "
                    f"succeeded={rc.succeeded}  errored={rc.errored}  "
                    f"processing={rc.processing}"
                )
                results_by_id = self._try_collect_anthropic_batch(batch_id)

        self._delete_batch_state(benchmark_name)
        return self._process_batch_results(
            items, results_by_id, answer_tag, benchmark_name, setting, timestamp
        )

    # ──────────────────────────────────────────────────────────────────────────
    # OpenAI: submit (non-blocking) + collect (non-blocking)
    # ──────────────────────────────────────────────────────────────────────────

    def _submit_openai_batch(self, items, benchmark_name: str, effective_prompt: Optional[str]) -> dict:
        """Upload requests and create a batch. Returns {batch_id, file_id} immediately."""
        import tempfile

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
        return {"batch_id": batch.id, "file_id": file_obj.id}

    def _try_collect_openai_batch(self, batch_id: str, file_id: str) -> Optional[dict]:
        """
        Non-blocking check.  Returns ``results_by_id`` dict when the batch is
        done, or ``None`` if it is still processing.
        Raises ``RuntimeError`` for failed/expired/cancelled batches.
        """
        batch = self.client.batches.retrieve(batch_id)
        terminal = {"completed", "failed", "expired", "cancelled"}
        if batch.status not in terminal:
            return None

        if batch.status != "completed":
            raise RuntimeError(
                f"OpenAI batch {batch_id} ended with status: {batch.status}"
            )

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

        try:
            self.client.files.delete(file_id)
        except Exception:
            pass

        return results_by_id

    # ──────────────────────────────────────────────────────────────────────────
    # Anthropic: submit (non-blocking) + collect (non-blocking)
    # ──────────────────────────────────────────────────────────────────────────

    def _submit_anthropic_batch(self, items, benchmark_name: str, effective_prompt: Optional[str]) -> str:
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

    def _try_collect_anthropic_batch(self, batch_id: str) -> Optional[dict]:
        """
        Non-blocking check.  Returns ``results_by_id`` dict when
        ``processing_status == "ended"``, or ``None`` otherwise.
        """
        batch = self.client.messages.batches.retrieve(batch_id)
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

    # ──────────────────────────────────────────────────────────────────────────
    # Shared result processing (same answer-extraction logic as VLLMEvaluator)
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

            # Strip <think> blocks
            thinking_trace = None
            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
            if think_match:
                thinking_trace = think_match.group(1).strip()
            response_for_answer = (
                re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
                if thinking_trace is not None else response
            )

            # Strip <reflection> blocks
            reflection = None
            ref_match = re.search(
                r"<reflection>(.*?)</reflection>", response_for_answer, re.DOTALL | re.IGNORECASE
            )
            if ref_match:
                reflection = ref_match.group(1).strip()

            # Extract answer using tag or first line
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

            expected = str(ground_truth).strip().lower()
            is_exact    = answer == expected
            is_contains = expected in answer
            is_prefix   = answer.startswith(expected)

            exact     += is_exact
            contains  += is_contains
            prefix_m  += is_prefix

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
