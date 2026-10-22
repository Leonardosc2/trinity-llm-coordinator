"""New synthetic tests of extracted functions, not historical benchmark runs."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from research.coordination import (
    AgentIdentity, BenchmarkExample, categorical_from_uniform, choose_batch,
    clean_extracted_answer, deterministic_episode_seed, extract_choice_answer,
    extract_final_answer, extract_last_boxed, extract_python_solution,
    format_transcript, run_episode, stable_softmax, verifier_accepts,
)
from research.statistics import wilson_interval


class SyntheticCoordinator:
    """Deterministic test fixture, not a trained coordinator."""
    agents = (AgentIdentity("synthetic", "Synthetic model"),)

    def __init__(self, roles):
        self.roles = iter(roles)

    def choose_action(self, state_text, rng):
        role_index = next(self.roles)
        return 0, role_index, {"model_probs": [1.0], "role_probs": [0.0, 0.0, 1.0]}


class SyntheticClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, agent_index, role, transcript, answer_contract):
        self.calls += 1
        return next(self.responses)


class ResearchTests(unittest.TestCase):
    def test_softmax_probability_sum(self):
        p = stable_softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertTrue(np.all(p > 0.0))

    def test_softmax_shift_invariance(self):
        a = np.array([-2.0, 0.0, 2.0])
        np.testing.assert_allclose(stable_softmax(a), stable_softmax(a + 10000.0))

    def test_categorical_intervals(self):
        p = np.array([0.2, 0.3, 0.5])
        self.assertEqual(categorical_from_uniform(p, 0.1), 0)
        self.assertEqual(categorical_from_uniform(p, 0.3), 1)
        self.assertEqual(categorical_from_uniform(p, 0.9), 2)

    def test_empty_transcript(self):
        value = format_transcript("synthetic task", [])
        self.assertIn("synthetic task", value)
        self.assertIn("no prior turns", value)

    def test_transcript_preserves_role(self):
        value = format_transcript("task", [{"turn":1,"model":"Synthetic model","role":"Worker","response":"draft"}])
        self.assertIn("Role: Worker", value)
        self.assertIn("draft", value)

    def test_accept_requires_first_nonempty_line(self):
        self.assertTrue(verifier_accepts("\nJUDGMENT: ACCEPT\nFINAL ANSWER: 7"))
        self.assertFalse(verifier_accepts("Explanation\nJUDGMENT: ACCEPT"))
        self.assertFalse(verifier_accepts("JUDGMENT: REVISE"))

    def test_worker_accept_text_does_not_terminate(self):
        client = SyntheticClient(["JUDGMENT: ACCEPT\nFINAL ANSWER: 7", "JUDGMENT: ACCEPT\nFINAL ANSWER: 7"])
        episode = run_episode(SyntheticCoordinator([1,2]), client, "task", "contract", 3, 4)
        self.assertEqual(client.calls, 2)
        self.assertTrue(episode["accepted"])

    def test_verifier_accept_terminates(self):
        client = SyntheticClient(["JUDGMENT: ACCEPT\nFINAL ANSWER: 7"])
        episode = run_episode(SyntheticCoordinator([2]), client, "task", "contract", 5, 4)
        self.assertEqual(len(episode["turns"]), 1)
        self.assertTrue(episode["accepted"])
        self.assertEqual(episode["predicted_answer"], "7")

    def test_turn_limit_returns_last_response(self):
        client = SyntheticClient(["FINAL ANSWER: 5", "FINAL ANSWER: 7"])
        episode = run_episode(SyntheticCoordinator([1,1]), client, "task", "contract", 2, 4)
        self.assertFalse(episode["accepted"])
        self.assertEqual(episode["predicted_answer"], "7")

    def test_last_final_answer(self):
        self.assertEqual(extract_final_answer("FINAL ANSWER: 5\nFINAL ANSWER: 7"), "7")

    def test_nested_boxed_answer(self):
        self.assertEqual(extract_last_boxed(r"\boxed{\frac{1}{2}}"), r"\frac{1}{2}")

    def test_cleanup_preserves_tuple(self):
        self.assertEqual(clean_extracted_answer("The final answer is (1, 2)."), "(1, 2)")

    def test_choice_answer(self):
        self.assertEqual(extract_choice_answer("FINAL ANSWER: C"), "C")

    def test_code_extraction_does_not_execute(self):
        self.assertEqual(extract_python_solution("```python\nprint('synthetic')\n```"), "print('synthetic')")

    def test_episode_seed_reproducibility(self):
        a = deterministic_episode_seed(7, 2, "math:synthetic", 0)
        self.assertEqual(a, deterministic_episode_seed(7, 2, "math:synthetic", 0))
        self.assertNotEqual(a, deterministic_episode_seed(7, 2, "math:synthetic", 1))

    def test_balanced_batch_reproducibility(self):
        pools = {name:[BenchmarkExample(name,str(i),"synthetic task","","synthetic","",{}) for i in range(6)] for name in ["math500","mmlu"]}
        first = choose_batch(pools, 2, 0, 7)
        second = choose_batch(pools, 2, 0, 7)
        self.assertEqual([x.key for x in first], [x.key for x in second])
        self.assertEqual(sum(x.benchmark == "math500" for x in first), 2)
        self.assertEqual(sum(x.benchmark == "mmlu" for x in first), 2)
        self.assertEqual(len({x.key for x in first}), 4)

    def test_wilson_interval_bounds(self):
        low, high = wilson_interval(16, 20)
        self.assertLess(low, 0.8)
        self.assertGreater(high, 0.8)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_reported_aggregate_consistency(self):
        path = Path(__file__).resolve().parents[1]/"results"/"reported_metrics.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.values():
            if isinstance(row,dict) and "paired_both_correct" in row:
                n = row["paired_both_correct"] + row["paired_both_incorrect"] + row["candidate_only_correct"] + row["baseline_only_correct"]
                self.assertEqual(n, row["mutually_scorable"])
                self.assertEqual(row["baseline_correct"],row["paired_both_correct"]+row["baseline_only_correct"])
                self.assertEqual(row["candidate_correct"],row["paired_both_correct"]+row["candidate_only_correct"])
        v5=data["v5_training"]
        self.assertEqual(sum(v5["role_selections"].values()),v5["worker_calls_returned"])
        self.assertEqual(sum(v5["verifier_outcomes"].values()),v5["role_selections"]["Verifier"])
        self.assertEqual(v5["accept_terminations"]+v5["forced_terminations"],v5["candidate_question_trajectories"])


if __name__ == "__main__":
    unittest.main()
