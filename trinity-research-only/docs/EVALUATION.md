# Evaluation protocol and methodological caveats

## What the supplied evaluators implement

The MATH-500 holdout evaluator compares each individual worker with a trained coordinator on a deterministic selection of questions. Its more developed revision excludes indices recorded in both completed-generation summaries and available episode logs, including interrupted generations. It rejects reuse of a saved holdout whose indices now overlap that recorded training history.

The later holdout revision also includes the selection seed in its output naming and restricts migration of prior results to the matching seed and current holdout. Earlier variants lack some of these safeguards. The corrected naming convention addresses one bookkeeping problem; it does not prove complete provenance or decontamination.

The multibench evaluator uses protected MATH-500 indices, the MMLU test split, an RLPR validation source, and LiveCodeBench V6. It compares workers and the coordinator on matched selections, saves results incrementally, and reports benchmark accuracy, error counts, timing, and Wilson intervals. Those are features of the recovered code, not a claim that every historical run was complete or valid.

## Limits of the implementation

The source's completed-record loader treats a recorded system/question key as completed even when the record contains an error. Resuming is therefore not automatically a retry policy. The result identity also does not fully bind every runtime, checkpoint content, grader, and generation setting. Do not reuse one output stream for changed experiments and assume old records are comparable.

The complete historic question-usage history is unavailable in this research-only release. Index exclusion can only cover the history actually supplied; missing records, different dataset orderings, and near-duplicates are separate limitations. Changing a seed alone does not prove a new sample is unseen.

The included Wilson interval function describes one system's proportion. It does not perform a paired significance test or account for adaptive selection of the best candidate. The paired test values in the historical research notes were reported for those specific comparisons; they have not been independently regraded here.

## Later evaluation practices reported in the notes

The research record separates training, validation, and held-out evaluation. It records symmetric handling of unscorable questions, distinguishes accepted-but-unscored cases from demonstrably false accepts, and compares identical items between baseline and candidate.

It also documents important qualifications:

- **v1:** a runtime failure in one arm made an apparent raw accuracy improvement disappear on mutually scorable items.
- **v2:** regrading saved traces corrected answer-format and exclusion handling. The pilot did not improve frozen-test accuracy and was not promoted.
- **v3 validation:** runtime repairs and context compaction occurred during evaluation. Retained episode costs omitted some earlier failed attempts. The aggregate difference cannot be attributed only to learned weights.
- **v3 held-out:** locked automatic scores and post-hoc format adjudication are different analyses of the same 37 questions. The latter must not silently replace the former.
- **v4 development:** tests and contamination checks were implementation checks, not evidence of improved model performance.
- **v5:** the reported perfect score is on optimization questions, not a held-out test.

## Integrity boundaries retained in this release

Previously evaluated held-out questions are treated as consumed, not reusable final tests for future tuning. Their text, identifiers, model outputs, and item-level adjudications are not included here.

The records recommend fixing the evaluation protocol and model-selection criteria before subsequent final evaluation, tracking all attempts and missing usage, and reporting accuracy alongside inference effort and verifier reliability. These recommendations are retained as research lessons, not presented as completed new experiments.

## Reproduction boundary

This release includes no trained checkpoints, original benchmark records, raw traces, model-service configuration, or full original experiment environment. It can reproduce the included helper-function tests and arithmetic on reported aggregate counts. It cannot reproduce or independently verify the historical model-performance results.
