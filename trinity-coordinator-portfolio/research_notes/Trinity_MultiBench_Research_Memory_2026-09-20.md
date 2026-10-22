# Trinity-Style Multi-Agent Coordinator: MultiBench Research Memory

**Canonical experiment record**  
**Experiment period:** September 18–20, 2026  
**Record finalized:** September 21, 2026

## 1. Research objective

Evaluate whether separable CMA-ES training of a small coordinator can improve
the accuracy–cost tradeoff of a multi-agent system across four distinct
benchmarks:

1. Math
2. MMLU
3. RLPR
4. LiveCodeBench

The main comparison is between the existing coordinator checkpoint and a new
one-generation MultiBench pilot trained from that checkpoint.

## 2. System and checkpoints

- Coordinator backbone: Qwen3-0.6B.
- Trainable component: a small coordination head optimized with separable
  CMA-ES.
- Main external agents observed in the runs: DeepSeek V4.1 Flash and Kimi K3.
- Local machine: Apple M4 MacBook Pro with 16 GB memory.
- Code-execution grading: isolated with Docker.
- Baseline checkpoint: `checkpoints/mac_sep_cma_recovery_v2`
- Pilot checkpoint directory: `checkpoints/multibench_v2_pilot`
- Pilot trained head: `checkpoints/multibench_v2_pilot/latest_head.npz`

## 3. Pilot training configuration

- Generations: 1
- Population: 4 candidates
- Questions per benchmark per candidate: 1
- Benchmarks: 4
- Total training episodes: 16
- Maximum model calls: 64
- Maximum turns per episode: 4
- Best reported training reward: 0.711
- Best reported training accuracy: 75.0%
- Starting checkpoint was read-only; the new head was saved separately.

This was deliberately a small pilot, not a full-scale training run.

## 4. Evaluation and grading repairs

Before drawing conclusions, the evaluation stack was repaired and verified.
The final grader revision used for the corrected analysis was
`multibench-v2.1.1`.

Important corrections:

- RLPR percentage answers are matched correctly when a reference stores
  percentages as plain numeric values but the candidate uses `%` signs.
- Explanatory values such as `100 g` and repeated gram quantities no longer
  cause correct percentage compositions to be rejected.
- Incorrect percentage values remain rejected.
- LiveCodeBench verifier acceptance checks published/public tests in Docker
  without exposing hidden tests or gold answers to the coordinator.
- Python solutions with unresolved runtime annotations, such as `List[int]`
  without an import, are rejected during preflight.
- `reference_error` and other unscored outcomes are not counted as ordinary
  incorrect answers or verifier false accepts.
- Paired comparison reports accuracy transitions, turns, reported tokens, and
  scorable false accepts.
- The final implementation passed all 13 unit tests.

These corrections were applied by regrading saved traces. Regrading made no
model calls and did not update weights.

## 5. Corrected validation comparison

The validation runs shared 20 questions, with 19 mutually scorable questions
because one RLPR item had a reference error.

| Metric | Baseline | Pilot | Change |
|---|---:|---:|---:|
| Mutually scored accuracy | 16/19 | 16/19 | Tie |
| Turns | 58 | 55 | -3 (-5.2%) |
| Reported tokens | 104,353 | 90,749 | -13,604 (-13.0%) |
| Scorable false accepts | 2 | 2 | No change |

### Validation benchmark breakdown

| Benchmark | Baseline | Pilot |
|---|---:|---:|
| Math | 5/5 | 5/5 |
| MMLU | 4/5 | 5/5 |
| RLPR | 4/4, 1 excluded | 4/4, 1 excluded |
| LiveCodeBench | 3/5 | 2/5 |

Paired behavior:

- Pilot-only win: one MMLU question.
- Baseline-only win: one LiveCodeBench question.
- Correct to correct: 15.
- Correct to incorrect: 1.
- Incorrect to correct: 1.
- Incorrect to incorrect: 2.
- Reference error to reference error: 1.

### Validation interpretation

The pilot matched baseline validation accuracy while using fewer turns and
substantially fewer reported tokens. However, it exchanged one LiveCodeBench
success for one MMLU success rather than producing a net accuracy gain.

## 6. Frozen-test comparison

Both checkpoints were evaluated once on the same frozen set using:

- Split: test
- Seed: 43
- Questions per benchmark: 5
- Maximum turns: 5
- Total paired questions: 20

### Frozen-test benchmark results

| Benchmark | Baseline | Pilot |
|---|---:|---:|
| Math | 5/5 | 5/5 |
| MMLU | 5/5 | 5/5 |
| RLPR | 2/5 | 1/5 |
| LiveCodeBench | 4/5 | 4/5 |
| **Overall** | **16/20 (80%)** | **15/20 (75%)** |

### Frozen-test cost and verifier results

| Metric | Baseline | Pilot | Change |
|---|---:|---:|---:|
| Turns | 64 | 60 | -4 (-6.25%) |
| Reported tokens | 113,476 | 111,792 | -1,684 (-1.48%) |
| Scorable false accepts | 3 | 4 | +1 |

Paired behavior:

- Pilot-only wins: none.
- Baseline-only win: `rlpr:23f5d59cdbb07a26c1ed`.
- Correct to correct: 15.
- Correct to incorrect: 1.
- Incorrect to incorrect: 4.

## 7. Final decision

**Retain the baseline checkpoint `checkpoints/mac_sep_cma_recovery_v2`.**

**Do not promote the pilot checkpoint `checkpoints/multibench_v2_pilot`.**

The pilot reduced turns slightly, but the frozen-test token saving was small,
accuracy fell by one question, and false accepts increased by one. The pilot
therefore did not improve the overall accuracy–cost–reliability tradeoff.

## 8. Main research finding

The one-generation CMA-ES pilot found a more economical coordination policy,
but the validation efficiency gain did not generalize fully to the frozen
test, and the policy did not improve accuracy. The observed test regression
was isolated to RLPR; Math, MMLU, and LiveCodeBench scores were unchanged.

Paper-ready statement:

> On a 20-question paired frozen evaluation across Math, MMLU, RLPR, and
> LiveCodeBench, the trained coordinator reduced model turns from 64 to 60
> (6.25%) and reported tokens from 113,476 to 111,792 (1.48%), but accuracy
> decreased from 16/20 to 15/20 and verifier false accepts increased from 3 to
> 4. The pilot was therefore not selected over the baseline.

## 9. What can and cannot be claimed

Supported conclusions:

- The training process changed coordinator behavior.
- The pilot showed an efficiency signal on validation and a smaller efficiency
  signal on the frozen test.
- The pilot did not improve frozen-test accuracy.
- RLPR selection or verification was the clearest remaining weakness.
- The corrected evaluation pipeline can measure benchmark-level accuracy,
  paired transitions, inference cost, and false accepts.

Unsupported conclusions:

- Do not claim that the pilot is generally worse across the entire benchmark
  distributions; the frozen test contained only 20 questions.
- Do not claim a statistically established accuracy difference from a single
  discordant item.
- Do not claim the validation token reduction generalized at the same size;
  it fell from 13.0% on validation to 1.48% on the frozen test.
- Do not describe the pilot as an accuracy improvement.

## 10. Research-integrity status

- The frozen test used here is now **consumed**.
- Do not tune rewards, routing, prompts, or graders specifically against its
  individual outcomes.
- Do not reuse this test result as an untouched final evaluation for a future
  model.
- Future development must use training and validation data only.
- A future v3 model requires a new untouched test sample or held-out seed for
  its final evaluation.

## 11. Recommended v3 direction

1. Increase the number and diversity of RLPR training and validation examples.
2. Use multiple fixed validation seeds rather than a single small sample.
3. Strengthen the reward penalty for verifier false accepts.
4. Preserve explicit cost penalties for excess turns and tokens.
5. Track per-benchmark reward so efficiency gains cannot hide a benchmark
   regression.
6. Predefine model-selection criteria before the next final test.
7. Evaluate the selected v3 checkpoint once on a new untouched paired test.

## 12. Canonical artifacts and logs

- Final repair package: `trinity-multibench-v2.1.1.zip`
- Corrected baseline validation:
  `logs/multibench_v2_baseline_validation_regraded_v211`
- Corrected pilot validation:
  `logs/multibench_v2_pilot_validation_regraded_v211`
- Baseline frozen test: `logs/multibench_v2_baseline_test_v211`
- Pilot frozen test: `logs/multibench_v2_pilot_test_v211`

This record should be treated as the canonical source for the MultiBench v2
pilot when preparing the eventual research paper.

## 13. Predeclared MultiBench v3 experiment

**Protocol added:** September 21, 2026, before v3 training or validation
results were observed.

The v3 experiment starts from the retained baseline
`checkpoints/mac_sep_cma_recovery_v2`; it does not warm-start from the rejected
v2 pilot.

### v3 hypothesis

A coordinator objective that explicitly protects the weakest benchmark and
more strongly penalizes false verifier acceptance can retain cross-benchmark
accuracy while reducing routing cost more reliably than the v2 mean-only
objective.

### v3 training changes

- Default RLPR sampling multiplier: 2.
- Default false-accept penalty: 0.75, increased from 0.40.
- Candidate fitness: 65% overall mean episode reward plus 35% of the
  candidate's weakest benchmark reward.
- Recommended run: 2 generations, population 4, two base questions per
  benchmark, RLPR multiplier 2, and maximum 4 turns.
- Recommended training budget: 80 episodes and at most 320 model calls.
- Training remains resumable and does not load validation or test data.

### v3 validation protocol

- Three deterministic, non-overlapping validation shards.
- Three questions per benchmark per shard.
- Baseline and candidate evaluated on identical questions.
- Total validation budget: 72 episodes across both checkpoints and at most 360
  model calls.
- Validation only; the consumed v2 test is not loaded.

### Predeclared promotion gate

The v3 checkpoint can advance only if it:

1. matches or exceeds baseline aggregate validation accuracy;
2. does not regress on any individual benchmark;
3. does not increase scorable verifier false accepts; and
4. reduces reported tokens by at least 3% when aggregate accuracy is tied.

If the gate returns `PROMOTE: NO`, the candidate will be archived and no new
test will be run. If it returns `PROMOTE: YES`, one new untouched paired test
sample will be defined and evaluated once.

### v3 implementation record

- Release package: `trinity-multibench-v3.zip`
- Training entry point: `train_multibench_v3.py`
- Paired validation entry point: `validate_multibench_v3.py`
- Standalone gate entry point: `select_multibench_v3.py`
- Automated regression suite: 17 tests passing.

## 14. Historical MultiBench v1 retrospective

**Archive reviewed:** `multibench-test-comparison(1).zip`

This is an earlier, pre-v2 experiment comparing the retained baseline
`checkpoints/mac_sep_cma_recovery_v2` with the historical trained checkpoint
`checkpoints/multibench_v1`. Both runs used the same 20-question test sample
(five questions from each benchmark), seed 43, and a maximum of five turns.

### Corrected interpretation

The original summaries reported 14/20 correct for the baseline and 15/20 for
the trained checkpoint. That one-question apparent gain is not a defensible
accuracy improvement: the baseline run contained one LiveCodeBench
`episode_error`, so only 19 questions were mutually scorable. On those 19
paired questions, the checkpoints tied at **14/19**.

The three non-identical outcomes were:

- `mmlu:293330d07957e72e9989`: `missing_final_answer -> correct`, a genuine
  trained-only win.
- `livecodebench:f79b942d330a7bf3a0b2`: `correct -> incorrect`, a genuine
  baseline-only win.
- `livecodebench:b0f25495a291a5efdd22`: `episode_error -> correct`; this is an
  infrastructure recovery, not a mutually scorable accuracy win.

Thus the mutually scorable win and loss cancel. This archive supplies no clean
evidence that the v1-trained router improved accuracy.

### Cost comparison

| Metric | Baseline | v1 trained | Change |
|---|---:|---:|---:|
| Correct on mutually scorable questions | 14/19 | 14/19 | tie |
| Total turns | 59 | 66 | +7 (+11.9%) |
| Reported tokens | 101,362 | 107,764 | +6,402 (+6.3%) |

The v1-trained checkpoint was therefore more expensive without improving fair
paired accuracy.

### Routing changes

| Routing signal | Baseline | v1 trained |
|---|---:|---:|
| DeepSeek calls | 26 | 30 |
| Kimi calls | 15 | 20 |
| Qwen3-4B calls | 15 | 10 |
| Qwen2.5-Coder calls | 3 | 4 |
| Local-model calls | 0 | 2 |
| Thinker-role turns | 5 | 11 |
| Verifier-role turns | 21 | 23 |
| Worker-role turns | 33 | 32 |
| Guardrail interventions | 21 | 28 |

The trained router shifted work toward DeepSeek and Kimi, doubled Thinker-role
usage, and triggered more guardrail interventions. That redistribution did not
produce a net paired accuracy gain.

### Benchmark-level token effects

| Benchmark | Baseline mean tokens | v1-trained mean tokens | Change |
|---|---:|---:|---:|
| Math | 4,940.8 | 3,646.0 | -26.2% |
| MMLU | 2,251.2 | 2,890.8 | +28.4% |
| RLPR | 3,687.4 | 5,128.4 | +39.1% |
| LiveCodeBench | 9,393.0 | 9,887.6 | +5.3% |

The historical training found a useful Math efficiency pattern but paid for it
with substantially higher MMLU and RLPR cost. This supports the v3 decision to
use worst-benchmark-aware fitness, explicit cost constraints, and a promotion
gate that rejects aggregate savings or gains that conceal benchmark-specific
regressions.

### Paper-use rule

Treat this as a historical ablation/diagnostic result, not as the paper's
primary trained-versus-baseline comparison. The archive predates the corrected
v2 grading and reporting rules. Its question sample is consumed and must not be
used for tuning or represented as an untouched final evaluation.

## 15. MultiBench v3 training and validation-resume incident

**Run date:** September 21, 2026

The predeclared v3 training run completed all two generations and four
candidates per generation. The saved checkpoint is
`checkpoints/multibench_v3/latest_head.npz`. Generation 2 reported a best
training reward of 0.109 and best training accuracy of 60.0%. These are
training-selection measurements, not validation or test results.

During the three-shard paired validation, the retained baseline reached an RLPR
row on validation shard 2 that the deterministic grader reported as
`grader_unavailable`. The original v3 evaluator treated this honest unscored
outcome like an infrastructure failure and paused the resumable run after
recording it.

The v3.0.1 validation-resume repair changes only failure handling:

- `grader_unavailable` remains recorded and is excluded from paired accuracy;
- a question is included in the gate only when both checkpoints receive a
  scorable outcome;
- episode/runtime failures and grader implementation errors still pause;
- completed episodes and the recorded unscored outcome remain untouched;
- training, model outputs, validation sampling, gate thresholds, and all
  test-split protections are unchanged.

The repaired regression suite passes 18 tests. The validation must resume in
the existing `logs/multibench_v3_validation` directory so previously completed
episodes are skipped. This is a protocol/infrastructure repair, not a tuning
change, and must be disclosed in the paper's evaluation methodology.

After installing v3.0.1, candidate validation shard 2 encountered a genuine
`episode_error` on LiveCodeBench item
`livecodebench:ba7e8dc0f8ad91b4bd9c`. Review found that the resume logic treated
every recorded result ID as completed, including transient failure records.
That behavior contradicted the evaluator's instruction to rerun the same
command after fixing an episode failure.

The v3.0.2 transient-resume repair therefore:

- removes only `episode_error`, `episode_interrupted`, and `grader_error`
  records when an incomplete shard resumes;
- reruns those exact question IDs while preserving every successful or
  honestly unscored episode;
- stores and prints a bounded exception message if a transient failure recurs;
- leaves routing, weights, questions, scoring, and the promotion gate unchanged.

The v3.0.2 regression suite passes 19 tests. This is also an infrastructure
repair rather than post-validation model tuning. The existing validation output
directory must be retained so only the failed episode is repeated.

On retry, the stored exception identified the failure precisely:
`ValueError: Conversation exceeds the local context budget.` Four long
DeepSeek turns on LiveCodeBench item `livecodebench:ba7e8dc0f8ad91b4bd9c`
caused the next local Qwen routing-state encoding to exceed its 8,192-token
limit.

The v3.0.3 coordinator-context repair preserves the full routing state whenever
it fits. Only after that exact overflow is detected, routing is retried with a
bounded view containing the question head and tail, every turn's routing
metadata, short answer excerpts, and a bounded candidate excerpt. The selected
agent still receives the normal full problem and shared prompt. Weights,
questions, grading, gate thresholds, and test protection remain unchanged.

The v3.0.3 regression suite passes 20 tests. Together with v3.0.2, the same
validation command retries only the failed episode and preserves all previously
completed episodes. This runtime robustness repair must be disclosed alongside
the earlier v3 validation infrastructure repairs.


## V3 validation archive audit — 2026-09-22

Source: `multibench-v3-validation-review.zip`, containing six completed shard summaries, 72 episode JSON files and selection.json. Archive SHA256: 81123ee539e35743750b123d153351dec21402b55b1e7ea8f253b8b52dbfe2c8.

Recomputed from the archive: 36 distinct question IDs per arm, with no within-arm shard overlap; matching paired IDs and data hash, validation split, seed 317, max_turns 5, and shard settings. Checkpoint paths differ as intended. All 72 summary records agree with their episode files for turn count, reported tokens, stop reason and verifier acceptance. Every retained turn has a total_tokens field and per-turn sums match episode totals. This verifies accounting consistency, not independent grading correctness or provider billing.

| Metric | Baseline | V3 candidate |
|---|---:|---:|
| Correct on mutually scored items | 25/34 (73.5%) | 28/34 (82.4%) |
| Math | 9/9 | 9/9 |
| MMLU | 6/9 | 7/9 |
| RLPR | 5/7 | 7/7 |
| LiveCodeBench | 5/9 | 5/9 |
| Scorable verifier false accepts | 4 | 1 |
| Retained episode turns | 112 | 118 |
| Reported tokens | 192,962 | 234,065 |

Observed accuracy gain: 3 questions, +8.82 percentage points. Tokens increase 41,103 (+21.30%); turns increase 6 (+5.36%). These are retained evaluation episode costs, not training cost, full experiment cost, or dollar cost. Earlier failed/retried attempts are not recoverable from this archive. No efficiency improvement is established.

Paired outcomes: 24 both correct, 5 both incorrect, 4 candidate-only correct, 1 baseline-only correct, 2 both unscored. Candidate-only wins: rlpr:31b7e236a57afe8966b9, rlpr:1944a9bd5f1067f32ba3, livecodebench:b31c96004bcc3f50ff2d, mmlu:1cf52f19499f92175599. Baseline-only win: livecodebench:78ab812776406e356a66. Exact two-sided paired binomial/McNemar test on the five discordant pairs: p=0.375. This small validation sample does not establish a statistically reliable general improvement, especially after model selection.

Exclusions are symmetric: rlpr:e5ba7041c2093a821383 is marked reference_error and rlpr:8595c2e195cbf9a89efb is marked grader_unavailable in both arms. The latter has the same textual final answer in both arms. Both excluded answers are verifier-accepted in both arms; report accepted-unscored separately from false accepts. The reference-error reason asserts a derivative/reference mismatch, but the source references are not in this archive. The grader-unavailable status alone does not prove a defective reference. The local RLPR grader can return this status depending on answer parseability. Independent reference review remains necessary. Across all 36 attempted items, observed correct counts are 25 and 28; the two unscored outcomes remain unknown.

Runtime amendment evidence: candidate episodes livecodebench:ba7e8dc0f8ad91b4bd9c and livecodebench:78ab812776406e356a66 each used context_compacted at turn 5; no baseline episode records compaction. The first is correct in both arms; the second is the candidate regression. Compaction changes the coordinator input and must be reported as a runtime amendment. Do not attribute the aggregate difference solely to trained weights. Code hashes and checkpoint content hashes are absent; path equality alone does not establish runtime/checkpoint identity.

Correction to earlier incident wording: v3.0.2 removes retryable records when loading a summary, without an incomplete-only condition, and replaces the failed episode file on rerun. It does not preserve full attempt history/cost. v3.0.3 uses character bounds, not a guaranteed tokenizer-level bound. Passing mocked unit tests did not establish complete end-to-end robustness. Earlier statements calling all repairs routing-neutral or exclusions objectively verified were too strong.

Decision: the configured validation gate reports PROMOTE:YES, and its headline arithmetic matches the archive. Treat this as selection for further evaluation, not product release or confirmed research superiority. Recommended next phase: freeze code, grader, checkpoint hashes and exclusion policy; retain all attempt costs; evaluate both arms on a predeclared disjoint held-out set. Exclude all previously used training/validation/test IDs explicitly—changing a seed does not guarantee unseen questions. Resolve grading validity independently before reporting final paper claims. Do not tune on held-out outcomes.

Paper-safe wording: In an exploratory paired validation evaluation, the V3 candidate answered 28 of 34 mutually scorable questions correctly versus 25 for the baseline, with scorable verifier false accepts decreasing from four to one. Reported token usage increased by 21.3%. Two questions were unscored in both arms. Runtime repairs occurred during evaluation, and the small sample does not establish general superiority.


## V3 holdout preparation and pre-inference review — 2026-09-22

Source: uploaded multibench_v3_holdout_preparation.zip. Local preparation reported
1,671 history files, 720 IDs and 965 normalized question hashes; no read errors
or missing checkpoint files. Available unused pools: Math 78, each other benchmark
95. All overlapping source hashes match the uploaded current-code archive.
Disjointness is limited to available local history and normalized exact text;
missing H20 history and near-duplicates remain limitations.

Before any new model answers, fixed three exclusions from the 40-row proposal:
interactive LCB b480490cbe8aa51ccb9b, malformed-table RLPR 45ef4f3f3196a7c8528f,
and price-tick-ambiguous RLPR cdad224ba579662aba18. No replacement sampling.
Retained sample: Math10, MMLU10, RLPR8, LCB9 =37 paired questions /74 episodes.

Discovered existing functional grader cannot decode newline-separated arguments
in three proposed LCB rows. Supplied patch preserves older encodings and adds
linewise JSON argument decoding. Eight retained RLPR rows use predeclared scalar
targets/tolerances; numeric-subset matching could otherwise accept incomplete or
incorrect fractions. Original datasets remain unchanged. These grader amendments
apply to both arms and make results a new protocol, not directly interchangeable
with historical scores. MMLU labels remain supplied benchmark labels; code tests
were decoded but not all independently validated with solution oracles.

Deliverable: trinity-v3-fixed-holdout.zip. Contains additive paired runner,
protocol with source/data/checkpoint hashes, numeric scoring, grader patch,
focused tests and review/methodology README. Default command performs hash and
Docker smoke checks without inference. Explicit --execute runs baseline and V3,
alternating arm order by question, max5 turns. Attempt files are unique, prior
failure costs are retained when available, missing usage and unknown-cost attempts
are flagged; dynamic unscored outcomes pause. Internal provider retries or calls
lost before saving may still be absent from reported costs. Four focused tests
passed locally; live Mac Docker/API execution remains unverified here. No new
accuracy results have been obtained. See package README for targets and exclusions.


## V3 fixed heldout final audit — 2026-09-22

Source: `multibench-v3-heldout-results.zip`. Archive SHA256:
`39c3028ecf3fd8ed57099da9fbca7b10e1e80e3ed20aa7e9670783c42519be34`.
The stored protocol SHA256 matches the frozen protocol:
`a30e5a816d6b09f36744eeff910d5c0cdfae752eb5634efba0d590992264c92e`.

The run is complete: 37 paired questions/74 episodes, with Math10, MMLU10,
RLPR8 and LiveCodeBench9 per arm. All arm/question keys are unique and paired;
`pending` is null; there are no unscored or unknown-cost attempts. Summary and
episode files agree on grades, verifier acceptance, stops, turns, and tokens.
Every retained turn has usage and episode token sums reconcile exactly. This
verifies archive/accounting consistency, not provider billing completeness.

Locked automatic scores: baseline 29/37 (78.4%) and V3 30/37 (81.1%). Paired
outcomes: 27 both correct, 5 both incorrect, 3 V3-only, 2 baseline-only; exact
two-sided paired binomial/McNemar p=1.0. By benchmark: Math 9-9, MMLU 9-8,
RLPR 4-4, LiveCodeBench 7-9 (baseline-V3). The locked +1 result is not evidence
of a reliable general gain.

Post-hoc review found symmetric strict-parser false negatives. Both arms gave
the correct “12th grade” for a Math reference stored as `12`, correct $453,120
cash flow embedded in prose, and correct approximately-632-year decay answers.
Baseline also gave the correct RLPR ship-rate expression/31.32 approximation.
Format-only adjudication therefore gives baseline 33/37 and V3 33/37 (89.2%
each): Math 10-10, MMLU 9-8, RLPR 7-6, LiveCodeBench 7-9. Paired outcomes become
31 both correct, 2 both incorrect, 2 V3-only and 2 baseline-only; exact p=1.0.
This is a post-hoc sensitivity result and must not replace the locked result
without disclosure. Both arms' `185 W` heater answer remains outside the frozen
185.10397±0.02 rule; accepting ordinary integer rounding would raise both arms
equally to 34/37 and would not change the comparison.

The format-adjudicated V3-only wins are LiveCodeBench b28ee2437faaa5ff30e1 and
fc2f8f993b0cf97ee386. Baseline-only wins are MMLU ad42afc5e8edf4c489b8 and
RLPR 6c31245b26bdd2903c17 (NPV: baseline $1,828.04 correct; V3 $1,827.45
incorrect). Format-adjudicated verifier false accepts are baseline 1 and V3 2;
V3 does not improve verifier safety on this holdout.

Retained cost: baseline 118 turns/205,883 reported tokens; V3 116
turns/201,185 tokens, a reduction of 2 turns (-1.7%) and 4,698 tokens (-2.3%).
This is not uniform: V3 tokens were -40.4% on Math, +4.1% on MMLU, -1.9% on
RLPR, and +6.1% on LiveCodeBench. Episode wall time fell 10.5% but is only a
descriptive latency observation. V3 shifted toward more DeepSeek and Thinker
turns and fewer Kimi/Coder/Verifier/Worker turns. Candidate LiveCodeBench item
fc2f8f993b0cf97ee386 used context compaction at turn 5 and was a V3-only win,
so that gain cannot be attributed solely to trained weights.

Conclusion for the paper: validation selection did not translate into a
confirmed heldout advantage. V3 shows a LiveCodeBench strength offset by MMLU
and RLPR regressions, an overall adjudicated tie, mixed benchmark-level cost,
and no false-accept improvement. Preserve this as the frozen final evaluation;
do not tune on these 37 questions or combine them with validation. Full audit:
`Trinity_V3_Heldout_Audit_2026-09-22.md`.


## V4 contamination-safe development preview — 2026-09-22

After freezing the V3 heldout evidence, development moved to a separate V4
workstream. No V3 heldout questions, outputs, grades, or adjudications were used
as code fixtures, prompt examples, routing targets, or model-selection criteria.
No model calls were made and no weights were changed.

The V4 development overlay adds:

- conservative terminal-scalar parsing that accepts common final-answer
  wrappers but rejects ambiguous multi-number spans;
- grading only against explicit, predeclared numeric targets and absolute
  tolerances, with percentage semantics enforced;
- context-aware integer-label normalization for cases such as a numeric
  reference versus a labelled ordinal, only when the question contains that
  label;
- a machine-checkable verifier protocol with mutually exclusive ACCEPT and
  REVISE schemas;
- exact candidate binding: an ACCEPT is invalid unless the verifier repeats the
  current candidate without changing it;
- a contamination scanner that accepts the frozen protocol as an external input
  and checks development artifacts for selected holdout IDs;
- a written policy prohibiting V3 heldout reuse for V4 tuning or promotion.

Verification: 30 unit tests passed (20 inherited plus 10 synthetic V4 tests),
all new Python modules compiled, the release ZIP passed archive integrity checks,
and the external holdout-ID scan reported zero violations across the release.
The synthetic cases do not copy the V3 heldout questions or answers.

Deliverable: `trinity-multibench-v4-dev.0.zip`. SHA256:
`af62d9ea42f5a309a6603f1c25662a7418ec80b579f1a663599711c32c041dff`.
This is a development overlay, not a trained checkpoint, product release, or new
research result. The next allowed evidence step is offline validation/development
checking, followed by code/prompt/rule freezing and a newly sampled untouched
holdout. The completed V3 fixed holdout must not be rerun to choose V4 changes.


## V4 development-evaluator safety gate — 2026-09-22

The V4 dev.0 overlay was independently installed on the Mac. Its combined
regression suite passed 30/30 tests, and the external V3 holdout-ID scan examined
20 files with `violations: []`. This verifies the intended installation and
exact-ID contamination check; it does not establish model quality, semantic
decontamination, or a product result.

Inspection then found that the inherited evaluation entry point still imported
coordinator V5. To prevent a paid development run from silently testing the
wrong runtime, dev.1 adds a separate V4 development evaluator and runtime loader
that explicitly use `multi_turn_coordinator_v6`. The evaluator is locked to the
already-consumed validation split, requires an external frozen-holdout protocol,
rejects any selected-ID overlap, records the protocol hash and runner identity,
and defaults to plan-only. Provider/model calls require the additional explicit
flag `--execute`.

The dev.1 regression suite passes 32 tests, including synthetic checks that
plan-only mode does not enter preflight/model execution and that holdout overlap
fails closed. All affected Python modules compile. The contamination scan reports
zero violations across 23 development files. The archive manifest verifies.

Deliverable: `trinity-multibench-v4-dev.1.zip`. SHA256:
`10fe25c7f414f526352a1b12d88cd6f549d3dfe334a1ff36cd291dd4c10f7955`.
No model calls were made, no benchmark scores were produced, and no weights were
updated in this step. The next allowed action is to inspect a small paired
validation-only plan before separately authorizing execution. The frozen V3
heldout remains prohibited for V4 development or model selection.
