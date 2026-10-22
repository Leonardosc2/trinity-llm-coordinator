# Trinity-Style Multi-Agent LLM Coordinator — Research Portfolio

**Status: partial recovery of a 2026 research project, not a fully runnable release.**

This repository documents my work experimenting with a lightweight **Qwen3-0.6B coordinator** that routes tasks to specialized language models. A compact coordination head was optimized using **separable CMA-ES**, with experiments on mathematical reasoning, knowledge, reasoning/problem-solving, and coding benchmarks. The project began on an **8× NVIDIA H20** server and continued on an **Apple M4 MacBook Pro** using a local/API model pool.

> This repository contains recovered scripts and experiment notes. Some original dependencies, agent services, datasets, grading modules, model weights, and checkpoints are not available here. Results are documented as historical experimental records, not freshly reproduced by this repository.

## Architecture

```text
Question + conversation history
              |
       Qwen3-0.6B coordinator
          + routing head
              |
     select model and role
       (Thinker / Worker / Verifier)
              |
       specialist language model
              |
         output / feedback
              |
    update history; stop / repeat
```

## Recoverable implementation

- [`src/train_sep_cma_multibench.py`](src/train_sep_cma_multibench.py) — historical training driver for four-benchmark separable CMA-ES optimization. May depend on missing project modules, local data, and running expert services.
- [`src/evaluate_multibench.py`](src/evaluate_multibench.py) — historical resumable benchmark evaluation driver. May depend on missing graders, datasets, and agent endpoints.
- [`research_notes/Trinity_MultiBench_Research_Memory_2026-09-20.md`](research_notes/Trinity_MultiBench_Research_Memory_2026-09-20.md) — detailed, corrected experiment record and integrity caveats.
- [`research_notes/V5_RESULTS.md`](research_notes/V5_RESULTS.md) — later local v5 pilot, separate from the multibench pilot below.
- [`research_notes/original_starter_README.md`](research_notes/original_starter_README.md) — description of an earlier starter design, **not evidence that all listed source modules were recovered**.

## Benchmark and training history

The original H20-server experiment explored **MATH-500**, **MMLU**, **RLPR**, and **LiveCodeBench**, training a compact coordination policy over a heterogeneous model pool including Qwen3-4B, Qwen2.5-Coder-3B, and DeepSeek-R1-Qwen-7B. Earlier notes report **66% on a MATH-500 coordinator evaluation**, compared with a **55% best single-agent baseline** in that specific run. The underlying raw logs/checkpoint have not been recovered in this package, so this number should not be treated as independently reproduced.

A later September 2026 local MultiBench pilot used DeepSeek V4.1 Flash and Kimi K3 as external workers and a one-generation / four-candidate CMA-ES experiment. Its corrected, paired evaluation gave:

| Metric | Baseline | Trained pilot |
|---|---:|---:|
| Validation accuracy (mutually scored) | 16/19 | 16/19 |
| Validation turns | 58 | 55 |
| Validation reported tokens | 104,353 | 90,749 |
| Frozen-test accuracy | 16/20 | 15/20 |
| Frozen-test turns | 64 | 60 |
| Frozen-test reported tokens | 113,476 | 111,792 |
| Frozen-test false accepts | 3 | 4 |

**Conclusion:** the pilot reduced some inference effort, but did **not** improve held-out accuracy, and was **not promoted**. This is a small experiment, not a statistically conclusive generalization claim. See the canonical research memory for grading repairs and integrity notes.

Another **separate** local v5 pilot completed 3 generations, 12 candidates, 240 training trajectories, and 1,194 worker calls. Its highest candidate scored 20/20 on **training questions only**; that must not be presented as held-out accuracy. The v5 notes document verifier failures and limited actual model/role collaboration.

## Research lessons

1. Routing policies need paired held-out evaluation, not just optimization-set reward.
2. A reduction in token use is not sufficient if accuracy or verification reliability worsens.
3. Benchmark graders and unscorable outcomes strongly affect conclusions.
4. Model selection diversity alone does not prove effective multi-agent collaboration.
5. Preserve a clean separation between training, validation, frozen test, and subsequent exploratory runs.

## What is missing

Original complete application modules, agent wrappers, exact Python environment, model weights, checkpoint files (`best_vector.npy`, `latest_head.npz`, etc.), actual datasets, original server logs, and API credentials are **not included**. Some file names and directory names were remembered from development rather than available as downloadable files. Do not assume this repository runs end-to-end without rebuilding those dependencies.

## Local privacy / publishing checklist

Before pushing, manually check that source code and notes do not contain private machine addresses, employer-proprietary implementation, sensitive paths, leaked evaluation sets, tokens, API keys, or material you are not authorized to publish. Do not upload paid benchmark data or model weights without checking licenses.

## Research context

This is an independent documentation of a Trinity-style coordinator reproduction/extension project, **not the official Trinity repository or an official replication of its reported performance**.
