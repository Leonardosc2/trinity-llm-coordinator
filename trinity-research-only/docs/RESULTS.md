# Reported experimental findings

## Evidence status

The following observations are transcribed from supplied research notes, not newly measured. Original traces and item-level reference material are excluded. The arithmetic can be checked from `results/reported_metrics.json`, but that does not independently validate grading, question provenance, or provider usage accounting.

The different versions below are separate experiments or development stages. Their sample counts must not be added together and reported as one independent test.

## v1: historical retrospective

The original raw summaries appeared to improve from 14/20 to 15/20. One baseline item was a runtime failure, however, leaving 19 mutually scorable questions. The fair paired comparison was **14/19 versus 14/19**. Turns rose from 59 to 66 and reported tokens from 101,362 to 107,764.

This retrospective did not establish an accuracy gain or an efficiency improvement. It predates the later grading corrections and is retained only as a diagnostic observation.

## v2: one-generation multi-benchmark pilot

The report describes four candidates, one training question per benchmark per candidate, 16 training episodes, and at most four turns per training episode. This was a small pilot rather than a full-scale training study.

| Metric | Baseline | Pilot |
|---|---:|---:|
| Validation, mutually scored correct | 16/19 | 16/19 |
| Validation turns | 58 | 55 |
| Validation reported tokens | 104,353 | 90,749 |
| Validation scorable false accepts | 2 | 2 |
| Frozen-test correct | 16/20 | 15/20 |
| Frozen-test turns | 64 | 60 |
| Frozen-test reported tokens | 113,476 | 111,792 |
| Frozen-test scorable false accepts | 3 | 4 |

One validation item was unscorable in both arms. The frozen test contained five questions per benchmark. Its per-benchmark correct counts were Math 5/5 versus 5/5, MMLU 5/5 versus 5/5, RLPR 2/5 versus 1/5, and LiveCodeBench 4/5 versus 4/5.

Reported token use fell about 13.0% on validation but only 1.48% on the frozen test. The pilot did not improve the observed accuracy-reliability tradeoff and was not promoted. A one-question regression does not establish general inferiority either.

## v3: validation selection and held-out evaluation

The documented development objective combined 65% overall mean episode reward and 35% weakest-benchmark reward, with stronger false-accept penalties and additional RLPR sampling. These settings belong to this later experiment, not automatically to the recovered earlier trainer.

### Exploratory validation

| Metric | Baseline | Candidate |
|---|---:|---:|
| Correct on mutually scorable questions | 25/34 | 28/34 |
| Scorable verifier false accepts | 4 | 1 |
| Retained turns | 112 | 118 |
| Reported tokens | 192,962 | 234,065 |

Two of 36 attempted questions per arm were unscored in both arms. Four questions were candidate-only successes and one was baseline-only. The research record reports a two-sided paired test p-value of 0.375; the small validation sample does not establish a reliable general gain. Reported token use increased 21.3%.

Runtime amendments included context compaction in candidate episodes. Failed-attempt costs were not fully preserved. Selection for further evaluation was not a demonstration of superiority, and the result cannot be attributed exclusively to changed weights.

### Held-out evaluation

The final reported sample contained 37 paired questions: Math 10, MMLU 10, RLPR 8, and LiveCodeBench 9.

| Scoring interpretation | Baseline | Candidate |
|---|---:|---:|
| Locked automatic scoring | 29/37 (78.4%) | 30/37 (81.1%) |
| Post-hoc format-only sensitivity analysis | 33/37 (89.2%) | 33/37 (89.2%) |

The locked paired outcomes were 27 both correct, 5 both incorrect, 3 candidate-only successes, and 2 baseline-only successes. The post-hoc interpretation had 31 both correct, 2 both incorrect, and 2 unique successes per arm. The record reports p = 1.0 for both paired comparisons.

Post-hoc format adjudication is explicitly not a new test and does not supersede the locked automatic scoring. It addressed reference-format mismatches, not a new model run. Specific questions, answers, identifiers, and internal review material are omitted from this release.

Retained turns were 118 versus 116; reported tokens were 205,883 versus 201,185, a reduction of about 2.3%. Format-adjudicated false accepts were 1 versus 2. Benchmark-specific changes were mixed. One candidate-only coding success used context compaction, so improvement on that item was not attributable solely to trained weights.

**Interpretation:** validation selection did not translate into a confirmed general held-out advantage. The post-hoc overall comparison tied, and verifier reliability did not improve on this sample.

## v4: development work, not a model result

The record describes more conservative numeric parsing, explicit reference tolerances, structured verifier judgments, candidate binding, and protection against reusing consumed held-out items. It reports development tests and contamination checks, but no new benchmark scores or weight updates at this stage.

The implementation and historical test suites are not included or rerun here. Development checks are not evidence of better model quality or complete semantic decontamination.

## v5: separate local-model optimization pilot

The separate v5 report covers three generations with four candidates each. Every candidate was evaluated on 20 optimization questions: 10 Math and 10 MMLU. Thus the report's 240 trajectories do **not** mean 240 unique questions.

| Generation | Candidate fitnesses | Mean | Best |
|---|---|---:|---:|
| 1 | 0.00, 0.60, 1.00, 0.00 | 0.400 | 1.00 |
| 2 | 0.60, 0.15, 0.95, 1.00 | 0.675 | 1.00 |
| 3 | 0.95, 0.30, 0.95, 0.20 | 0.600 | 0.95 |

The first candidate reaching maximum fitness was selected; another later candidate tied it. **20/20 is a training-selection score**, vulnerable to selection overfit, not a generalization estimate.

| Coordination diagnostic | Reported count |
|---|---:|
| Attempted / returned worker calls | 1,194 / 1,194 |
| Thinker selections | 17 |
| Worker selections | 903 |
| Verifier selections | 274 |
| Trajectories using multiple models | 42/240 |
| Trajectories using multiple roles | 16/240 |
| Valid verifier ACCEPT judgments | 4 |
| Valid verifier REVISE judgments | 245 |
| Malformed verifier judgments | 25 |
| Acceptance terminations | 4 |
| Forced terminations | 236 |

The report also lists five truncated-output flags and 45 missing-final-answer flags. These are call-level diagnostics that may overlap, not additional trajectory counts.

The supplied v5 report provides no held-out accuracy. It records successful frozen-base verification for its own implementation, but the underlying checkpoints and verification outputs are not included in this release. These observations cannot be treated as independent verification of another version's training mechanism.

## Conclusions supported by the records

The project demonstrates an implemented research approach to small-model coordination and a series of diagnostic evaluations. The strongest supported lessons concern evaluation quality: separate training and test performance, account for unscorable items, report protocol amendments, and measure accuracy, cost proxies, and verifier behavior together.

The supplied evidence does not establish a broadly superior learned coordinator, production readiness, or statistically reliable performance improvements. Earlier isolated accuracy claims without matching evidence in this release are not used as headline results.
