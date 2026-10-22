# TRINITY-Style Multi-Agent LLM Coordination

**Research-only portfolio: methods, selected code excerpts, and reported experimental findings.**

This project investigates whether a small language-model coordinator can choose **which model acts next** and **which role it takes** to improve the accuracy, inference effort, and reliability of a multi-model system.

The recovered training implementation uses **Qwen3-0.6B**, a compact model-and-role routing head, and **separable CMA-ES**. Its benchmark support includes **MATH-500, MMLU, RLPR, and LiveCodeBench**. The supplied research records also describe later multi-benchmark evaluations and a separate local-model training pilot.

> This is a TRINITY-inspired reproduction and experimentation project, not the official TRINITY implementation. All performance numbers below are **reported historical observations** from supplied research records, not results freshly reproduced by this release. Different experiments use different protocols and must not be combined into one benchmark score.

## Research question

Can a compact learned coordinator allocate reasoning, solving, and verification across heterogeneous language models more effectively than an existing coordination policy?

This is **multi-turn model-and-role selection**, not merely selecting one model for a single prompt.

```text
Question + accumulated transcript
                |
         Small coordinator
                |
       Model-and-role logits
                |
    Select a model and one role
     Thinker / Worker / Verifier
                |
       Generate the next response
                |
          Update transcript
                |
  Verifier ACCEPT or turn limit? --- no ---> repeat
                |
               yes
                |
          Terminal response
```

## Method

The recovered multibench trainer takes a hidden representation from the penultimate position of the completed coordinator sequence and applies a linear head with **L + 3 outputs**, where L is the worker-model count. Separate softmax distributions select a model and a role. The coordinator's generated text is discarded; its hidden state drives the routing decision.

The trainer optimizes the routing head and, optionally, log-scale adjustments to singular values of seven linear matrices in the coordinator's second-to-last transformer layer. Separable CMA-ES uses benchmark-graded terminal outcomes rather than backpropagating through agent responses.

In that trainer, candidates receive balanced benchmark batches and fitness is the negative of macro-averaged accuracy because the optimizer minimizes. Later experiments described in the research notes use additional cost and verifier-reliability criteria; they are not assumed to share this exact objective or implementation.

See [Methods](docs/METHODS.md) and [Evaluation](docs/EVALUATION.md).

## Reported results

### Multi-benchmark experiments

| Experiment | Baseline correct | Candidate correct | Interpretation |
|---|---:|---:|---|
| v2 corrected validation | 16/19 | 16/19 | Accuracy tie; fewer reported tokens |
| v2 frozen test | 16/20 | 15/20 | No accuracy gain; pilot not promoted |
| v3 exploratory validation | 25/34 | 28/34 | Observed gain, but higher token use and runtime amendments |
| v3 held-out, locked automatic scoring | 29/37 | 30/37 | One-question gain; not evidence of reliable superiority |
| Same v3 held-out questions, post-hoc format adjudication | 33/37 | 33/37 | Sensitivity analysis only; must not replace the locked scores |

For the **v2 frozen test**, turns decreased from 64 to 60 and reported tokens from 113,476 to 111,792, while correct answers decreased by one and scorable verifier false accepts increased from 3 to 4. The research record retained the baseline.

For the **v3 held-out evaluation**, reported tokens decreased from 205,883 to 201,185, but format-adjudicated accuracy tied and format-adjudicated false accepts increased from 1 to 2. The reported results do not establish a general accuracy or reliability advantage.

These are small paired experiments. Reported tokens describe recorded usage, not independently verified billing or complete training cost. The post-hoc row is the **same evaluation**, not another dataset or additional sample.

### Separate v5 optimization pilot

The supplied v5 report describes **3 generations, 12 candidates, 20 optimization questions per candidate, 240 candidate-question trajectories, and 1,194 returned worker calls**. Its selected candidate achieved **20/20 on training questions only**.

That is not held-out accuracy. Mean candidate fitness was 0.400, 0.675, then 0.600; there was no sustained generation-by-generation improvement. Only 42/240 trajectories used multiple models and 16/240 used multiple roles, indicating limited observed collaboration rather than a consistently collaborative policy.

See [Results and limitations](docs/RESULTS.md) for the separate histories and their caveats. [Aggregate data](results/reported_metrics.json) contains no question text, item identifiers, model outputs, or checkpoint contents.

## Research contributions documented

- Model-and-role coordination with a small backbone and compact routing parameters.
- Gradient-free optimization over benchmark outcomes.
- Paired evaluation against individual agents or an existing coordinator, depending on the experiment.
- Analysis of answer-extraction failures, verifier false accepts, unscorable outcomes, and resume bookkeeping.
- Explicit separation of optimization scores, validation-based selection, locked test results, and post-hoc interpretation.

## Included code

This release contains **selected deployment-free excerpts**, not the original training application:

| File | Scope |
|---|---|
| `research/coordination.py` | Sampling, transcript construction, role-based episode loop, balanced batches, answer extraction |
| `research/statistics.py` | Wilson interval function recovered from a holdout evaluator |
| `tests/test_research.py` | New synthetic unit tests for the included excerpts |
| `tools/summarize_results.py` | New arithmetic-only summary of the transcribed aggregate results |

Model loading, service clients, the full CMA-ES driver, production graders, and operational configuration are intentionally not included. The episode loop accepts caller-supplied interfaces. Its tests use synthetic responses; they neither contact a model nor reproduce benchmark results.

## Inspect the research excerpts

From the repository root, use an existing Python 3.10+ environment. NumPy is the only dependency of the included excerpts.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/summarize_results.py
```

These commands run synthetic checks and summary arithmetic only. They do **not** train a coordinator, retrieve a dataset, make model calls, or execute generated solutions. The dependency file is not a reconstruction of the original experiment environment.

## Limitations

The original experiment traces, full data provenance, exact model environments, and trained artifacts are not part of this research-only release. Consequently, the historical numbers cannot be independently reproduced or regraded from this repository alone. Passing the included unit tests establishes only behavior on those synthetic cases.

The evaluation notes document protocol changes, small samples, incomplete historical decontamination evidence, and post-hoc grading interpretation. Neither the high training fitness nor the small observed validation gains establish a reliable general advantage.

## Release scope

Personal and organizational identifiers, machine details, access information, deployment instructions, raw logs, raw datasets, screenshots, original archives, and trained binary artifacts are intentionally excluded. Only an explicitly selected set of research text and code is included; there is no prior repository history in this package.

See [Provenance and transformations](docs/PROVENANCE.md), [Publication scope](docs/PUBLICATION_SCOPE.md), and [Validation record](docs/VALIDATION.md). This release does not change the sharing status or history of any existing repository. Removing operational information does not determine permission to publish research or source code; no new license is assigned here.
