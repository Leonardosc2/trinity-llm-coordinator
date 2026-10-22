# Research provenance and transformations

## Supplied material used

The research description was grounded in the supplied multibench trainer, its earlier MATH-500 trainer, the holdout-evaluator revisions, the multibench evaluator, the consolidated MultiBench research record, and a separate v5 results report.

This release deliberately does not retain original archive names, personal paths, run-directory inventories, timestamps, question identifiers, source-archive fingerprints, or raw operational notes. The version labels v1 through v5 are retained to distinguish the experiments, not to imply one uniformly integrated codebase.

## Included source excerpts

The following bodies in `research/coordination.py` were copied from the supplied multibench trainer:

```text
BenchmarkExample
stable_softmax
categorical_from_uniform
format_transcript
verifier_accepts
run_episode
extract_balanced_braces
extract_last_boxed
extract_final_answer
clean_extracted_answer
extract_choice_answer
extract_python_solution
deterministic_episode_seed
choose_batch
```

The `ROLES` constant retains the original Thinker, Worker, and Verifier order. The imports are restricted to those needed by these excerpts.

For `run_episode`, the two integration type annotations were changed to `CoordinatorProtocol` and `AgentClientProtocol`. Its function body is otherwise preserved. Those Protocols and the two-field `AgentIdentity` data class are newly added interface declarations, not recovered historical training components. They contain no connection settings.

`research/statistics.py` preserves the `wilson_interval` function body from the more developed supplied holdout evaluator. The two uploaded copies of that evaluator with the same content were not treated as independent implementations.

The excerpted functions preserve historical behavior, including limitations. They do not silently fix grading, retry handling, bounds validation, context truncation, or data provenance. Synthetic tests are narrow checks, not an assertion of complete correctness.

## Newly prepared material

The README and documentation are new research-only summaries of the supplied material. The aggregate JSON is a transcription of reported counts. The summary script performs arithmetic on those counts; the new tests use synthetic fixtures only.

These new tests are separate from test totals mentioned in historical notes. Neither historical training nor historical evaluation was executed during preparation. No new model-quality result is claimed.

## Intentionally not included

No full model loaders, inference clients, service configurations, training launchers, credential-handling code, original logs, source datasets, generated-code graders, checkpoint loaders, or binary artifacts are included. This is a narrowly scoped research portfolio, not a drop-in sanitized version of the original application.

## Attribution and permission

The project is described as TRINITY-inspired because that is how the supplied implementation identifies itself. It is not official TRINITY source code, an author-endorsed reproduction, or a replication of the original paper's performance.

No new license or ownership claim is assigned. Review publication and redistribution permission separately before releasing source or unpublished research. Removal of operational identifiers cannot settle those permissions.
