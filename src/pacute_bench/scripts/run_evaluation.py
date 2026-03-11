#!/usr/bin/env python3
"""
Run benchmark evaluation against a running vLLM server.

Reads model configurations from configs/models_pt.yaml and configs/models_it.yaml,
then evaluates each requested model on the requested benchmarks using the vLLM
OpenAI-compatible API.

Usage:
    # Quick smoke-test (single model, 10 samples per benchmark)
    python scripts/run_evaluation.py \\
        --models gpt2 \\
        --max-samples 10 \\
        --vllm-url http://localhost:8000

    # Full evaluation run
    python scripts/run_evaluation.py \\
        --models qwen-2.5-7b-it \\
        --vllm-url http://localhost:8000 \\
        --vllm-model-id Qwen/Qwen2.5-7B-Instruct

    # Submit via PBS (recommended):
    bash scripts/submit_evaluations.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

from pacute_bench.evaluators import make_evaluator, BENCHMARK_FORMATS


# ---------------------------------------------------------------------------
# Model config helpers
# ---------------------------------------------------------------------------

def load_model_configs(config_paths=None) -> dict:
    """
    Load model configurations from one or more YAML files.

    Returns:
        Dict mapping model name → (hf_path, type, tokenizer, thinking, provider, reasoning_parser)
    """
    if config_paths is None:
        config_paths = [
            Path("configs") / "models_pt.yaml",
            Path("configs") / "models_it.yaml",
            Path("configs") / "models_commercial.yaml",
        ]
    elif isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]

    model_configs: dict = {}
    for cp in config_paths:
        cp = Path(cp)
        if not cp.exists():
            print(f"Warning: config file not found: {cp}")
            continue
        with open(cp) as f:
            data = yaml.safe_load(f)
        for name, info in data.get("models", {}).items():
            model_configs[name] = (
                info["path"],
                info["type"],
                info.get("tokenizer", info["path"]),
                info.get("thinking", False),
                info.get("provider", None),   # None → vLLM, "openai"/"anthropic" → CommercialEvaluator
                info.get("reasoning_parser", None),  # None → Qwen3-style enable_thinking kwarg
            )
    return model_configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_bench_summary(bench: str, results: dict) -> None:
    print(f"\n  {bench} Results  (n={results['num_samples']})")
    if results.get("format") == "generative":
        print(f"    Exact match  : {results['exact_match']:.4f}")
        print(f"    Contains     : {results['contains_match']:.4f}")
        print(f"    Prefix       : {results['prefix_match']:.4f}")
    else:
        print(f"    Accuracy     : {results['accuracy']:.4f}")
        print(f"    F1           : {results['f1_score']:.4f}")
        print(f"    Norm. Acc.   : {results['normalized_accuracy']:.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate models on pacute-bench benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-config",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Path(s) to model config YAML(s) (default: configs/models_pt.yaml + models_it.yaml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2"],
        help="Model keys to evaluate (must appear in the config YAMLs)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "pacute-affixation-mcq",
            "pacute-composition-mcq",
            "pacute-manipulation-mcq",
            "pacute-syllabification-mcq",
            "hierarchical-mcq",
            "langgame-mcq",
            "multi-digit-addition-mcq",
            "cute-gen",
            "pacute-affixation-gen",
            "pacute-composition-gen",
            "pacute-manipulation-gen",
            "pacute-syllabification-gen",
            "hierarchical-gen",
            "langgame-gen",
            "multi-digit-addition-gen",
        ],
        help="Benchmark names to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap samples per benchmark (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/benchmark_evaluation",
        help="Directory for combined results JSON (default: results/benchmark_evaluation)",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["auto", "mcq", "gen", "both"],
        default="auto",
        help=(
            "Evaluation mode. 'auto': MCQ-only for PT models, both for IT models. "
            "'mcq': MCQ only. 'gen': generative only. 'both': both for all models."
        ),
    )
    parser.add_argument(
        "--eval-config",
        default=None,
        help="Path to evaluation config YAML (default: configs/evaluation.yaml)",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Override system prompt for all generative benchmarks",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run benchmarks even when inference results already exist",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="Base URL of the running vLLM server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--vllm-api-key",
        default="token-abc123",
        help="API key for the vLLM server (default: token-abc123)",
    )
    parser.add_argument(
        "--vllm-model-id",
        default=None,
        help="Override the model ID used in API calls (useful when different from HF path)",
    )
    args = parser.parse_args()

    # ---- load eval config (system prompts, answer tags) --------------------
    eval_config_path = args.eval_config
    if eval_config_path is None:
        default_cfg = Path("configs") / "evaluation.yaml"
        if default_cfg.exists():
            eval_config_path = str(default_cfg)

    benchmark_system_prompts: dict = {}
    benchmark_answer_tags: dict    = {}
    if eval_config_path:
        with open(eval_config_path) as f:
            eval_cfg = yaml.safe_load(f)
        for bench, entry in eval_cfg.get("generative_instructions", {}).items():
            if isinstance(entry, dict):
                benchmark_system_prompts[bench] = entry.get("instruction", "")
                if "answer_tag" in entry:
                    benchmark_answer_tags[bench] = entry["answer_tag"]
            else:
                benchmark_system_prompts[bench] = str(entry)
        print(f"Loaded eval config: {eval_config_path}")
        print(f"  Instructions for: {list(benchmark_system_prompts)}")

    # ---- load model configs -------------------------------------------------
    model_configs = load_model_configs(args.model_config)
    if not model_configs:
        print("ERROR: No model configurations found — check configs/models_*.yaml")
        sys.exit(1)

    invalid = [m for m in args.models if m not in model_configs]
    if invalid:
        print(f"ERROR: Unknown model(s): {invalid}")
        print(f"Available: {list(model_configs)}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def filter_benchmarks(benchmarks: list[str], mode: str) -> list[str]:
        kept = []
        for b in benchmarks:
            fmt = BENCHMARK_FORMATS.get(b, "both")
            if mode == "both" or fmt == "both" or fmt == mode:
                kept.append(b)
            else:
                print(f"  Skipping {b} ({fmt}-only, effective mode={mode})")
        return kept

    print(f"\n{'='*80}")
    print("Evaluation Configuration")
    print(f"{'='*80}")
    print(f"  Eval mode  : {args.eval_mode}")
    print(f"  Overwrite  : {args.overwrite}")
    print(f"  vLLM URL   : {args.vllm_url}")
    print(f"{'='*80}\n")

    all_results: dict = {}

    for model_name in args.models:
        hf_path, model_type, tokenizer_name, thinking, provider, reasoning_parser = model_configs[model_name]

        if provider:
            effective_mode = "gen"   # commercial batch APIs: gen-only
        else:
            effective_mode = ("mcq" if model_type == "pt" else "both") \
                if args.eval_mode == "auto" else args.eval_mode

        benchmarks_for_model = filter_benchmarks(args.benchmarks, effective_mode)
        if thinking:
            thinking_label = f"thinking=ON ({reasoning_parser or 'enable_thinking kwarg'})"
        else:
            thinking_label = "thinking=OFF"

        print(f"\n{'='*80}")
        print(f"Model: {model_name}  (type={model_type}, mode={effective_mode}, {thinking_label})")
        print(f"Benchmarks: {', '.join(benchmarks_for_model)}")
        print(f"{'='*80}")

        if not benchmarks_for_model:
            print("No benchmarks to run — skipping.")
            continue

        try:
            if provider:
                evaluator = make_evaluator(
                    provider,
                    model_name=model_name,
                    model_id=hf_path,
                    thinking=thinking,
                    system_prompt=args.system_prompt,
                    benchmark_system_prompts=benchmark_system_prompts,
                    benchmark_answer_tags=benchmark_answer_tags,
                )
            else:
                evaluator = make_evaluator(
                    None,
                    model_name=model_name,
                    model_id=hf_path,
                    model_type=model_type,
                    tokenizer_name=tokenizer_name,
                    thinking=thinking,
                    reasoning_parser=reasoning_parser,
                    vllm_url=args.vllm_url,
                    vllm_api_key=args.vllm_api_key,
                    vllm_model_id=args.vllm_model_id,
                    system_prompt=args.system_prompt,
                    benchmark_system_prompts=benchmark_system_prompts,
                    benchmark_answer_tags=benchmark_answer_tags,
                )

            model_results: dict = {}

            if provider:
                # Commercial models: submit all batches at once, poll concurrently.
                bench_results = evaluator.evaluate_benchmarks_parallel(
                    benchmarks_for_model,
                    max_samples=args.max_samples,
                    check_existing=not args.overwrite,
                    timestamp=timestamp,
                )
                for bench, results in bench_results.items():
                    if not results or results.get("skipped"):
                        continue
                    detailed = results.pop("detailed_results", None)
                    results.pop("setting", None)
                    if detailed:
                        inf_dir = Path("results") / model_name / "inference"
                        inf_dir.mkdir(parents=True, exist_ok=True)
                        inf_file = inf_dir / f"{bench}.jsonl"
                        detailed.sort(key=lambda r: r.get("id", ""))
                        with open(inf_file, "w", encoding="utf-8") as f:
                            for r in detailed:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        print(f"  Saved inference results → {inf_file}")
                    model_results[bench] = results
                    _print_bench_summary(bench, results)
            else:
                # vLLM models: sequential (server handles one model at a time).
                for bench in benchmarks_for_model:
                    try:
                        res = evaluator.evaluate_benchmark(
                            benchmark_name=bench,
                            max_samples=args.max_samples,
                            check_existing=not args.overwrite,
                            timestamp=timestamp,
                        )
                    except Exception as e:
                        import traceback
                        print(f"ERROR on {bench}: {e}")
                        traceback.print_exc()
                        continue
                    if not res or res.get("skipped"):
                        continue
                    detailed = res.pop("detailed_results", None)
                    res.pop("setting", None)
                    if detailed:
                        inf_dir = Path("results") / model_name / "inference"
                        inf_dir.mkdir(parents=True, exist_ok=True)
                        inf_file = inf_dir / f"{bench}.jsonl"
                        detailed.sort(key=lambda r: r.get("id", ""))
                        with open(inf_file, "w", encoding="utf-8") as f:
                            for r in detailed:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        print(f"  Saved inference results → {inf_file}")
                    model_results[bench] = res
                    _print_bench_summary(bench, res)

            all_results[model_name] = {
                "hf_model_name": hf_path,
                "model_type":    model_type,
                "thinking":      thinking,
                "benchmarks":    model_results,
            }

            # Save per-model JSON
            model_dir = Path("results") / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_file = model_dir / f"evaluation_results_{timestamp}.json"
            with open(model_file, "w", encoding="utf-8") as f:
                json.dump({**all_results[model_name], "timestamp": timestamp},
                          f, indent=2, ensure_ascii=False)
            print(f"\n  Results saved → {model_file}")

            del evaluator

        except Exception as e:
            import traceback
            print(f"ERROR evaluating {model_name}: {e}")
            traceback.print_exc()

    # ---- combined results ---------------------------------------------------
    combined_file = Path(args.output_dir) / f"evaluation_results_{timestamp}.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*80}")
    print(f"Combined results → {combined_file}")
    print(f"{'='*80}")

    # ---- summary table ------------------------------------------------------
    print("\nSummary:")
    print(f"{'Model':<28} {'Benchmark':<28} {'Format':<8} {'Primary Metric'}")
    print("-" * 90)
    for mname, mdata in all_results.items():
        for bname, res in mdata["benchmarks"].items():
            fmt = res.get("format", "mcq")
            if fmt == "generative":
                metric = f"EM={res['exact_match']:.3f}"
            else:
                metric = f"Acc={res['accuracy']:.3f}  F1={res['f1_score']:.3f}"
            print(f"{mname:<28} {bname:<28} {fmt:<8} {metric}")
    print()


if __name__ == "__main__":
    main()
