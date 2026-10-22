# Faithful Trinity Local Pilot v5 Results

## Run completion and integrity

The v5 run completed normally according to its saved progress and checkpoint records. `progress.json` reports `status: complete`, 12/12 candidates, 3/3 generations, 240/240 candidate-question trajectories, and 1,194 attempted/1,194 returned worker calls (maximum budget: 1,200). The candidate event ledger contains 12 `candidate_complete` events and three `generation_complete` events. All 12 candidate files are marked complete and contain 20 question outcomes each.

A separate `completion.json` is not present. Completion is instead recorded in `progress.json` and corroborated by the event ledger and final files. The selected training checkpoint is `checkpoints/selected_training_candidate.json` (global candidate 2, training fitness 1.0); the optimizer checkpoint is `checkpoints/sep_cmaes_state.json` (generation 3). The selected checkpoint’s configuration hash matches the run configuration, and its optimizer-checkpoint hash matches the saved optimizer checkpoint. The sealed config, prompt, selection, and grader-label files all match their manifest SHA-256 values.

## Candidate fitness

Fitness is the fraction of 20 training questions receiving terminal reward 1. Math and MMLU each contribute 10 questions.

| Generation | Candidate in generation | Global candidate | Fitness | Math reward | MMLU reward |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 0.00 | 0/10 | 0/10 |
| 1 | 2 | 1 | 0.60 | 5/10 | 7/10 |
| 1 | 3 | 2 | **1.00** | **10/10** | **10/10** |
| 1 | 4 | 3 | 0.00 | 0/10 | 0/10 |
| 2 | 1 | 4 | 0.60 | 5/10 | 7/10 |
| 2 | 2 | 5 | 0.15 | 0/10 | 3/10 |
| 2 | 3 | 6 | 0.95 | 9/10 | 10/10 |
| 2 | 4 | 7 | **1.00** | **10/10** | **10/10** |
| 3 | 1 | 8 | 0.95 | 9/10 | 10/10 |
| 3 | 2 | 9 | 0.30 | 2/10 | 4/10 |
| 3 | 3 | 10 | 0.95 | 9/10 | 10/10 |
| 3 | 4 | 11 | 0.20 | 0/10 | 4/10 |

The selected checkpoint is global candidate 2, the earliest candidate attaining the maximum fitness of 1.0; global candidate 7 also attained 1.0. The selection rule in the runner breaks fitness ties by lower global candidate index. The selected candidate’s training scores are Math 10/10 and MMLU 10/10. These are optimization-set results, not held-out estimates.

## Fitness by generation

| Generation | Candidate fitnesses | Mean fitness | Best fitness |
|---:|---|---:|---:|
| 1 | 0.00, 0.60, 1.00, 0.00 | 0.400 | 1.00 |
| 2 | 0.60, 0.15, 0.95, 1.00 | 0.675 | 1.00 |
| 3 | 0.95, 0.30, 0.95, 0.20 | 0.600 | 0.95 |

Mean fitness rose from generation 1 to generation 2, then declined in generation 3. Best fitness did not improve beyond the initial generation’s 1.00 and ended below it. Thus the run does not show a sustained generation-over-generation improvement, although later candidates sometimes scored highly.

## Worker and role selections

Counts below cover all 1,194 worker calls across all candidates and turns.

| Worker | Calls | Share |
|---|---:|---:|
| Local-Qwen3-0.6B | 301 | 25.2% |
| `qwen3:4b-instruct` | 470 | 39.4% |
| `qwen2.5-coder:3b` | 423 | 35.4% |

| Role | Selections | Share |
|---|---:|---:|
| Thinker | 17 | 1.4% |
| Worker | 903 | 75.6% |
| Verifier | 274 | 22.9% |

The worker choice did not collapse to one model, but role use strongly favored Worker; Thinker was rare. Only 42/240 trajectories used more than one distinct worker, and only 16/240 used more than one distinct role. This is limited evidence of multi-turn coordination rather than a robust, consistently collaborative policy.

## Verifier and termination outcomes

Across the 274 selected Verifier turns, saved judgments were:

| Verifier result | Count |
|---|---:|
| Valid `ACCEPT` | 4 |
| Valid `REVISE` | 245 |
| Malformed/invalid judgment | 25 |

Four trajectories stopped on Verifier acceptance. The other 236 reached forced five-turn termination. Trajectory lengths were 237 at five turns and three at three turns.

## Output failures

Across worker-call event records, there were five truncated outputs and 45 missing-`FINAL_ANSWER` flags. There were 1,194 returned calls in total. The truncation and missing-final-answer counts are call-level flags and can overlap; they are not additional trajectory counts.

## Runtime and call totals

| Measure | Result |
|---|---:|
| Worker calls attempted / returned | 1,194 / 1,194 |
| Maximum call budget | 1,200 |
| Total recorded worker latency | 3,348.74 seconds |
| Mean worker-call latency | 2.81 seconds |
| Mean calls per trajectory | 4.98 |
| Mean summed worker latency per trajectory | 13.95 seconds |
| Wall-clock runtime (run start to final progress update) | 3,864.97 seconds (64 min 25 sec) |
| Output tokens recorded | 51,759 |

Wall-clock runtime exceeds summed model-call latency by about 516 seconds, reflecting process setup and non-generation work. This is a 20-question, 12-candidate training run; it is not a benchmark of user-facing response latency.

## Frozen base verification

The final progress record states `base_frozen_verified: true`. The runner snapshots every Qwen3-0.6B state-dict tensor before execution and calls `verify_frozen` after candidate evaluation; that check compares SHA-256 hashes for every tensor and raises if any changed. The run therefore records successful verification that the base weights remained unchanged. The selected coordinator head and authorized singular-value scales are separate learned parameters.

## Is held-out evaluation justified?

Yes, as a bounded generalization check, not as a claim of success. The training search selected a perfect 20/20 candidate from 12 candidates on only 20 optimization questions, and another candidate tied it. The result is especially vulnerable to selection overfit; the absence of sustained best-fitness improvement reinforces that concern. Freeze the selected checkpoint and evaluate it once on a newly sealed, permitted development set that was not used in this pilot. Do not use these 20 questions to tune it further, and do not access protected, locked-test, or frozen-holdout data.
