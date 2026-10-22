#!/usr/bin/env python3
"""Evaluate individual agents and a trained coordinator on unseen benchmarks.

Evaluation sets:
- protected MATH-500 holdout indices;
- MMLU official test split;
- RLPR WebInstruct verified validation split;
- LiveCodeBench V6.

Results are appended to JSONL and can be resumed safely.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch


EVALUATOR_VERSION = "2026-07-31-v1"
BENCHMARKS = ("math500", "mmlu", "rlpr", "livecodebench")


def import_trainer(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Trainer not found: {path}")
    spec = importlib.util.spec_from_file_location("trinity_multibench_trainer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    home = Path.home()
    project = home / "tingmin_trinity"
    parser = argparse.ArgumentParser(
        description="Unseen evaluation for the local TRINITY multibench checkpoint"
    )
    parser.add_argument(
        "--trainer",
        default=str(home / "train_sep_cma_multibench.py"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            project
            / "checkpoints/all4_local_tokens1024/best_coordinator.pt"
        ),
    )
    parser.add_argument(
        "--math_data",
        default=str(project / "data/math500_ready.jsonl"),
    )
    parser.add_argument(
        "--math_holdout_indices",
        default=str(
            project
            / "checkpoints/sep_cma_math500_tmux/"
            "evaluation_holdout_indices_seed20260731_n100.json"
        ),
    )
    parser.add_argument(
        "--mmlu_test",
        default=str(project / "data/mmlu_test.parquet"),
    )
    parser.add_argument(
        "--rlpr_eval",
        default=str(project / "data/rlpr_webinstruct_val.parquet"),
    )
    parser.add_argument(
        "--lcb_v6",
        default=str(project / "data/livecodebench_v6.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(
            project / "checkpoints/all4_local_tokens1024/evaluation_unseen"
        ),
    )
    parser.add_argument(
        "--num_per_benchmark",
        type=int,
        default=25,
        help="Use 25 for a same-day preliminary evaluation; 100 is stronger.",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_turns", type=int, default=0)
    parser.add_argument("--agent_max_tokens", type=int, default=1024)
    parser.add_argument("--agent_timeout", type=int, default=600)
    parser.add_argument("--lcb_timeout", type=int, default=6)
    parser.add_argument(
        "--systems",
        default="all",
        help="all, trinity, or comma-separated agent keys",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "vector" not in payload:
        raise ValueError(f"Unsupported checkpoint: {path}")
    return payload


def load_index_list(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, Mapping):
        values = next(
            (
                payload[key]
                for key in ("indices", "holdout_indices", "selected_indices")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
        if values is None:
            raise ValueError(f"No indices list found in {path}")
    else:
        raise ValueError(f"Invalid indices file: {path}")
    return [int(value) for value in values]


def load_lcb_v6(
    trainer: Any,
    path: Path,
    timeout: int,
) -> tuple[list[Any], Any]:
    try:
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    except ImportError as exc:
        raise RuntimeError(
            "LiveCodeBench is unavailable. Set PYTHONPATH to its repository."
        ) from exc

    examples: list[Any] = []
    samples: dict[str, dict[str, Any]] = {}
    for source_index, record in enumerate(trainer.iter_local_records(path)):
        problem = CodeGenerationProblem(**dict(record))
        external_id = str(
            getattr(problem, "question_id", None)
            or getattr(problem, "contest_id", None)
            or source_index
        )
        uid = f"{source_index}:{external_id}"
        question = str(
            getattr(problem, "question_content", None)
            or getattr(problem, "question", "")
        ).strip()
        starter_code = str(getattr(problem, "starter_code", "") or "").strip()
        if starter_code:
            question += f"\n\nSTARTER CODE:\n```python\n{starter_code}\n```"
        if not question:
            continue
        examples.append(
            trainer.BenchmarkExample(
                benchmark="livecodebench",
                uid=uid,
                question=question,
                gold="<hidden tests>",
                grader="livecodebench",
                answer_contract=(
                    "Return the complete Python 3 solution in one fenced ```python "
                    "code block. Do not add a FINAL ANSWER line."
                ),
                metadata={"source_index": source_index, "release": "v6"},
            )
        )
        samples[uid] = problem.get_evaluation_sample()
    if not examples:
        raise RuntimeError(f"No LiveCodeBench V6 examples loaded from {path}")
    return examples, trainer.LiveCodeBenchGrader(samples, timeout)


def deterministic_sample(
    examples: list[Any],
    count: int,
    seed: int,
    benchmark_position: int,
) -> list[Any]:
    if len(examples) < count:
        raise ValueError(
            f"{examples[0].benchmark if examples else 'dataset'} has "
            f"{len(examples)} examples; requested {count}"
        )
    rng = np.random.default_rng(seed + 1009 * benchmark_position)
    indices = sorted(
        int(index)
        for index in rng.choice(len(examples), size=count, replace=False)
    )
    return [examples[index] for index in indices]


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
        / denominator
    )
    return center - margin, center + margin


def load_completed(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    completed: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (
                    str(record["system"]),
                    str(record["benchmark"]),
                    str(record["example_uid"]),
                )
                completed[key] = record
            except (json.JSONDecodeError, KeyError, TypeError):
                print(
                    f"[warning] ignoring invalid result line {line_number}",
                    flush=True,
                )
    return completed


def choose_systems(
    value: str,
    agents: tuple[Any, ...],
) -> list[tuple[str, Optional[int]]]:
    available = {agent.key: index for index, agent in enumerate(agents)}
    if value.strip().lower() == "all":
        selected = [
            (f"agent:{agent.key}", index) for index, agent in enumerate(agents)
        ]
        selected.append(("trinity", None))
        return selected
    names = [item.strip() for item in value.split(",") if item.strip()]
    selected: list[tuple[str, Optional[int]]] = []
    for name in names:
        if name == "trinity":
            selected.append(("trinity", None))
        elif name in available:
            selected.append((f"agent:{name}", available[name]))
        else:
            raise ValueError(
                f"Unknown system {name!r}; use trinity or one of {sorted(available)}"
            )
    if not selected:
        raise ValueError("--systems cannot be empty")
    return selected


def main() -> int:
    args = parse_args()
    if args.num_per_benchmark <= 0:
        raise ValueError("--num_per_benchmark must be positive")

    trainer_path = Path(args.trainer).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = import_trainer(trainer_path)
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint.get("metadata") or {}
    train_args = metadata.get("args") or {}
    best_generation = int(metadata.get("generation", -1))

    print(f"[evaluator] {EVALUATOR_VERSION}", flush=True)
    print(f"[trainer] {trainer_path}", flush=True)
    print(f"[checkpoint] {checkpoint_path}", flush=True)
    print(f"[checkpoint best generation] {best_generation}", flush=True)

    math_all = trainer.load_math500(
        Path(args.math_data).expanduser().resolve(),
        limit=None,
        excluded_indices=set(),
    )
    math_by_index = {
        int(example.metadata["source_index"]): example for example in math_all
    }
    holdout_indices = load_index_list(
        Path(args.math_holdout_indices).expanduser().resolve()
    )
    missing_math = [index for index in holdout_indices if index not in math_by_index]
    if missing_math:
        raise RuntimeError(f"Missing MATH holdout indices: {missing_math[:10]}")
    math_holdout = [math_by_index[index] for index in holdout_indices]

    mmlu, _ = trainer.load_mmlu(
        "local",
        "all",
        "test",
        None,
        Path(args.mmlu_test).expanduser().resolve(),
    )
    rlpr, _ = trainer.load_rlpr(
        "local",
        "validation",
        None,
        Path(args.rlpr_eval).expanduser().resolve(),
    )
    lcb, lcb_grader = load_lcb_v6(
        trainer,
        Path(args.lcb_v6).expanduser().resolve(),
        args.lcb_timeout,
    )

    pools = {
        "math500": math_holdout,
        "mmlu": mmlu,
        "rlpr": rlpr,
        "livecodebench": lcb,
    }
    selected: dict[str, list[Any]] = {}
    selection_manifest: dict[str, list[str]] = {}
    for position, benchmark in enumerate(BENCHMARKS):
        chosen = deterministic_sample(
            pools[benchmark],
            args.num_per_benchmark,
            args.seed,
            position,
        )
        selected[benchmark] = chosen
        selection_manifest[benchmark] = [example.uid for example in chosen]

    manifest_path = output_dir / (
        f"selection_seed{args.seed}_n{args.num_per_benchmark}.json"
    )
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != selection_manifest:
            raise RuntimeError(
                f"Saved evaluation selection does not match: {manifest_path}"
            )
    else:
        manifest_path.write_text(
            json.dumps(selection_manifest, indent=2),
            encoding="utf-8",
        )

    raw_agents = checkpoint.get("agents") or [
        {
            "key": agent.key,
            "display_name": agent.display_name,
            "base_url": agent.base_url,
            "served_model_name": agent.served_model_name,
        }
        for agent in trainer.DEFAULT_AGENTS
    ]
    agents = tuple(trainer.AgentSpec(**item) for item in raw_agents)
    client = trainer.VLLMAgentClient(
        agents,
        timeout_seconds=args.agent_timeout,
        max_tokens=args.agent_max_tokens,
        temperature=0.0,
        retries=2,
    )
    client.healthcheck()

    coordinator_model = str(
        Path(
            train_args.get(
                "coordinator_model",
                Path.home() / "tingmin_trinity/models/Qwen3-0.6B",
            )
        ).expanduser()
    )
    max_turns = args.max_turns or int(train_args.get("max_turns", 5))
    coordinator = trainer.TrinityCoordinator(
        model_path=coordinator_model,
        agents=agents,
        device=args.device,
        max_input_tokens=int(
            train_args.get("coordinator_max_input_tokens", 4096)
        ),
        coordinator_new_tokens=int(train_args.get("coordinator_new_tokens", 8)),
        use_svd_tuning=int(checkpoint.get("svd_parameter_count", 0)) > 0,
        seed=int(train_args.get("seed", 42)),
    )
    vector = checkpoint["vector"]
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().cpu().numpy()
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (coordinator.vector_dim,):
        raise ValueError(
            f"Checkpoint vector shape {vector.shape}; expected "
            f"{(coordinator.vector_dim,)}"
        )
    coordinator.set_vector(vector)
    systems = choose_systems(args.systems, agents)

    result_path = output_dir / (
        f"results_seed{args.seed}_n{args.num_per_benchmark}.jsonl"
    )
    summary_path = output_dir / (
        f"summary_seed{args.seed}_n{args.num_per_benchmark}.json"
    )
    completed = load_completed(result_path)
    if completed:
        print(f"[resume] {len(completed)} completed evaluations", flush=True)

    with result_path.open("a", encoding="utf-8") as output:
        for system_name, agent_index in systems:
            for benchmark in BENCHMARKS:
                examples = selected[benchmark]
                print("\n" + "=" * 78)
                print(
                    f"EVALUATING {system_name} ON {benchmark} "
                    f"({len(examples)} questions)",
                    flush=True,
                )
                print("=" * 78)
                for position, example in enumerate(examples, start=1):
                    key = (system_name, benchmark, example.uid)
                    if key in completed:
                        continue
                    started = time.time()
                    error = None
                    episode: dict[str, Any]
                    predicted = ""
                    grader_details: Any = None
                    try:
                        if agent_index is None:
                            episode = trainer.run_episode(
                                coordinator=coordinator,
                                client=client,
                                question=example.question,
                                answer_contract=example.answer_contract,
                                max_turns=max_turns,
                                seed=trainer.deterministic_episode_seed(
                                    args.seed,
                                    0,
                                    example.key,
                                    0,
                                ),
                                verbose=False,
                            )
                        else:
                            state = trainer.format_transcript(example.question, [])
                            response = client.complete(
                                agent_index,
                                "Worker",
                                state,
                                example.answer_contract,
                            )
                            episode = {
                                "turns": [],
                                "accepted": False,
                                "final_text": response,
                                "predicted_answer": trainer.extract_final_answer(
                                    response
                                ),
                            }
                        reward, predicted, grader_details = trainer.score_example(
                            example,
                            episode,
                            lcb_grader,
                        )
                    except Exception as exc:  # noqa: BLE001
                        episode = {
                            "turns": [],
                            "accepted": False,
                            "final_text": "",
                            "predicted_answer": "",
                        }
                        reward = 0
                        error = f"{type(exc).__name__}: {exc}"

                    record = {
                        "system": system_name,
                        "benchmark": benchmark,
                        "example_uid": example.uid,
                        "position": position,
                        "reward": int(reward),
                        "gold": example.gold,
                        "predicted": predicted,
                        "final_text": episode["final_text"],
                        "turns": episode["turns"],
                        "accepted": episode["accepted"],
                        "grader_details": trainer.make_json_safe(grader_details),
                        "seconds": round(time.time() - started, 3),
                        "error": error,
                        "evaluation_seed": args.seed,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_best_generation": best_generation,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    completed[key] = record

                    current = [
                        item
                        for (saved_system, saved_benchmark, _), item
                        in completed.items()
                        if saved_system == system_name
                        and saved_benchmark == benchmark
                        and item["example_uid"]
                        in set(selection_manifest[benchmark])
                    ]
                    correct = sum(int(item["reward"]) for item in current)
                    print(
                        f"[{position:03d}/{len(examples)}] "
                        f"reward={reward} | running={correct}/{len(current)} "
                        f"({100.0 * correct / len(current):.1f}%)",
                        flush=True,
                    )
                    if error:
                        print(f"    [error] {error}", flush=True)

    summaries: dict[str, Any] = {}
    for system_name, _ in systems:
        by_benchmark: dict[str, Any] = {}
        benchmark_accuracies: list[float] = []
        for benchmark in BENCHMARKS:
            selected_uids = set(selection_manifest[benchmark])
            records = [
                item
                for (saved_system, saved_benchmark, uid), item
                in completed.items()
                if saved_system == system_name
                and saved_benchmark == benchmark
                and uid in selected_uids
            ]
            correct = sum(int(item["reward"]) for item in records)
            total = len(records)
            accuracy = correct / total if total else 0.0
            low, high = wilson_interval(correct, total)
            by_benchmark[benchmark] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "wilson_95_low": low,
                "wilson_95_high": high,
                "errors": sum(bool(item.get("error")) for item in records),
                "average_seconds": (
                    sum(float(item["seconds"]) for item in records) / total
                    if total
                    else 0.0
                ),
            }
            benchmark_accuracies.append(accuracy)
        summaries[system_name] = {
            "benchmarks": by_benchmark,
            "macro_average": float(np.mean(benchmark_accuracies)),
        }

    final_summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "checkpoint": str(checkpoint_path),
        "checkpoint_best_generation": best_generation,
        "seed": args.seed,
        "num_per_benchmark": args.num_per_benchmark,
        "selection_manifest": str(manifest_path),
        "systems": summaries,
    }
    summary_path.write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("FINAL UNSEEN RESULTS")
    print("=" * 78)
    for system_name, stats in summaries.items():
        pieces = []
        for benchmark in BENCHMARKS:
            result = stats["benchmarks"][benchmark]
            pieces.append(
                f"{benchmark}={result['correct']}/{result['total']} "
                f"({100.0 * result['accuracy']:.1f}%)"
            )
        print(
            f"{system_name}: " + " | ".join(pieces)
            + f" | macro={100.0 * stats['macro_average']:.1f}%",
            flush=True,
        )
    print(f"\nResults: {result_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
