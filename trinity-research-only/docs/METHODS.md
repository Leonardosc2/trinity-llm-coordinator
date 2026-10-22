# Method: model-and-role coordination

## Scope

This description follows the supplied multibench training script. The separate later pilots are described in the results record; their complete implementations are not reconstructed here. Details below describe the recovered implementation rather than asserting equivalence to the original TRINITY paper.

## State and roles

At each coordination turn, the state consists of the original task and the accumulated transcript. Each prior turn carries a model label, role, and response. The three roles are:

- **Thinker:** propose strategy, diagnose mistakes, and identify missing subgoals.
- **Worker:** make concrete progress toward a solution.
- **Verifier:** check the candidate and emit an acceptance or revision judgment.

The recovered loop stops only when the selected role is Verifier and the first nonempty response line matches `JUDGMENT: ACCEPT`, or when the turn budget is exhausted. The last response is returned. It has no separately learned STOP action. Syntactically valid acceptance is not the same as correctness; a later benchmark grader supplies the outcome.

## Routing representation

The recovered implementation loads a Qwen3-0.6B backbone, generates a short deterministic continuation, reruns the resulting sequence, and selects the final-layer hidden representation at the penultimate sequence position. Generated coordinator text itself is not sent to the agents.

For hidden vector h and L available workers, the linear head computes:

```text
z = W_head h + b_head          dimension: L + 3
p_model = softmax(z[0:L])
p_role  = softmax(z[L:L+3])
```

Model and role choices use two draws from the episode's seeded random generator. Thus the head represents separate model and role distributions rather than an unconstrained joint distribution over all model-role pairs.

The transcript builder assembles the full available history, but the backbone input is subject to a configured token limit. This is not a guarantee that arbitrarily long history is always retained in the model's encoded input. Later evaluation notes explicitly report context-compaction amendments.

## Evolvable parameters

The candidate vector concatenates the head's flattened weight matrix, head bias, and optional log singular-value scales.

For an adapted linear matrix with initial factorization W = U diag(s) Vh:

```text
W(a) = U diag(s * exp(a)) Vh
```

U and Vh are fixed. The recovered implementation bounds each log scale a to [-0.5, 0.5] and permits head parameters in [-1, 1]. Seven matrices in the second-to-last transformer layer are targeted: attention query, key, value, and output projections, plus the feed-forward gate, up, and down projections.

The code disables gradients and uses black-box fitness. When this SVD adaptation is enabled, selected backbone matrices are reconstructed from the scales; it would be misleading to describe every effective backbone matrix as unchanged. A separate v5 report describes its own frozen-base verification mechanism; that report is not proof that these historical implementations are identical.

## Optimization and batching

The recovered trainer uses separable CMA-ES. For each generation it draws a balanced batch from the enabled benchmark pools. All candidates in that generation receive the same chosen examples. Episode seeds are derived from the base seed, generation, example key, and replication index.

Within each benchmark, binary graded rewards are averaged. Benchmark means are then averaged with equal weight:

```text
score = mean_over_benchmarks(mean_of_binary_rewards)
optimizer_fitness = -score
```

This prevents a larger dataset from automatically dominating that batch's objective. It does not establish representativeness of the sampled questions. The script stores a best observed batch score across changing generation batches, so that value is a search statistic, not an unbiased held-out estimate.

The trainer catches episode exceptions, records them, and assigns zero reward. Later paired evaluation records distinguish model mistakes from unscored or failed episodes; the distinction must be made explicit when comparing experiments.

## Benchmark contracts in the recovered trainer

| Benchmark | Expected response | Grading approach in the source |
|---|---|---|
| MATH-500 | Terminal answer | Answer extraction and symbolic/numeric comparison |
| MMLU | One choice letter | Choice extraction and reference comparison |
| RLPR | Task-specific final answer | Rule-based matching of structured, numeric, or textual references |
| LiveCodeBench | Complete Python solution | Execution-based evaluation using benchmark tests |

Only answer-extraction helpers are included as source excerpts. Symbolic evaluation and generated-code execution are deliberately excluded from this release. Their absence is not a claim that the original graders were independently verified.
