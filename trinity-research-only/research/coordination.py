"""Research excerpts: routing, transcript construction, and answer extraction.

Derived from the supplied multibench trainer. Function bodies are preserved;
run_episode's two integration annotations use new Protocols instead of the
original deployment-specific classes. No model loader or service client is
included. See docs/PROVENANCE.md for the exact scope and limitations.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np

ROLES = ("Thinker", "Worker", "Verifier")


@dataclass(frozen=True)
class AgentIdentity:
    """Non-operational model labels needed by the recovered episode loop."""
    key: str
    display_name: str


class CoordinatorProtocol(Protocol):
    """Caller-supplied coordinator; no trained implementation is bundled."""
    agents: Sequence[AgentIdentity]

    def choose_action(
        self, state_text: str, rng: np.random.Generator
    ) -> tuple[int, int, dict[str, list[float]]]: ...


class AgentClientProtocol(Protocol):
    """Caller-supplied text completion interface; no connection configuration."""

    def complete(
        self, agent_index: int, role: str, transcript: str, answer_contract: str
    ) -> str: ...


@dataclass(frozen=True)
class BenchmarkExample:
    benchmark: str
    uid: str
    question: str
    gold: str
    grader: str
    answer_contract: str
    metadata: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.benchmark}:{self.uid}"


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - np.max(values)
    exp_values = np.exp(np.clip(shifted, -60.0, 60.0))
    total = float(exp_values.sum())
    if not math.isfinite(total) or total <= 0:
        return np.full(values.shape, 1.0 / len(values), dtype=np.float64)
    return exp_values / total


def categorical_from_uniform(probs: np.ndarray, uniform: float) -> int:
    cumulative = np.cumsum(probs)
    index = int(np.searchsorted(cumulative, uniform, side="right"))
    return min(index, len(probs) - 1)


def format_transcript(question: str, turns: Sequence[dict[str, Any]]) -> str:
    parts = ["ORIGINAL TASK:\n" + question.strip()]
    if not turns:
        parts.append("COLLABORATION TRANSCRIPT:\n(no prior turns)")
    else:
        lines = ["COLLABORATION TRANSCRIPT:"]
        for turn in turns:
            lines.append(
                f"\n--- Turn {turn['turn']} | Model: {turn['model']} | Role: {turn['role']} ---\n"
                f"{turn['response']}"
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def verifier_accepts(text: str) -> bool:
    first_nonempty = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return bool(re.fullmatch(r"JUDGMENT\s*:\s*ACCEPT\s*", first_nonempty, re.I))


def run_episode(
    coordinator: CoordinatorProtocol,
    client: AgentClientProtocol,
    question: str,
    answer_contract: str,
    max_turns: int,
    seed: int,
    verbose: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    turns: list[dict[str, Any]] = []
    accepted = False

    for turn_number in range(1, max_turns + 1):
        state = format_transcript(question, turns)
        model_index, role_index, probabilities = coordinator.choose_action(state, rng)
        role = ROLES[role_index]
        agent = coordinator.agents[model_index]
        response = client.complete(
            model_index,
            role,
            state,
            answer_contract=answer_contract,
        )
        turn = {
            "turn": turn_number,
            "model": agent.display_name,
            "model_key": agent.key,
            "role": role,
            "response": response,
            **probabilities,
        }
        turns.append(turn)
        if verbose:
            print(
                f"      turn {turn_number}: {agent.display_name} + {role}",
                flush=True,
            )
        if role == "Verifier" and verifier_accepts(response):
            accepted = True
            break

    final_text = turns[-1]["response"] if turns else ""
    return {
        "turns": turns,
        "accepted": accepted,
        "final_text": final_text,
        "predicted_answer": extract_final_answer(final_text),
    }


def extract_balanced_braces(text: str, start: int) -> Optional[str]:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    return None


def extract_last_boxed(text: str) -> Optional[str]:
    positions = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    for position in reversed(positions):
        brace = text.find("{", position)
        if brace >= 0:
            value = extract_balanced_braces(text, brace)
            if value is not None:
                return value.strip()
    return None


def extract_final_answer(text: str) -> str:
    matches = re.findall(r"FINAL\s+ANSWER\s*:\s*(.+)", text, flags=re.I)
    if matches:
        answer = matches[-1].strip()
        return clean_extracted_answer(answer)
    boxed = extract_last_boxed(text)
    if boxed:
        return clean_extracted_answer(boxed)
    # Last non-empty line is a conservative fallback.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return clean_extracted_answer(lines[-1]) if lines else ""


def clean_extracted_answer(text: str) -> str:
    """Remove harmless answer wrappers without changing the answer itself."""
    value = text.strip().strip("`").strip()
    value = re.sub(r"^(?:therefore|thus|hence)\s*[:,]?\s*", "", value, flags=re.I)
    value = re.sub(
        r"^(?:the\s+)?final\s+answer\s+(?:is|equals)\s*[:,]?\s*",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"\s*[,;—-]?\s*(?:this|that)\s+is\s+(?:the\s+)?(?:final\s+)?answer"
        r"\s*[.!]?\s*$",
        "",
        value,
        flags=re.I,
    )
    return value.rstrip(". ").strip()


def extract_choice_answer(text: str) -> str:
    """Extract an MMLU option letter while avoiding letters inside prose."""
    final = extract_final_answer(text)
    patterns = (
        r"^\s*[\(\[]?([A-D])[\)\].,:;]?\s*$",
        r"\b(?:answer|option|choice)\s*(?:is|:|=)?\s*[\(\[]?([A-D])\b",
        r"^\s*[\(\[]?([A-D])[\)\].,:;]\s+",
    )
    for candidate in (final, text):
        for pattern in patterns:
            matches = re.findall(pattern, candidate, flags=re.I | re.M)
            if matches:
                return str(matches[-1]).upper()
    return ""


def extract_python_solution(text: str) -> str:
    """Return the last fenced code block, or the complete response as fallback."""
    blocks = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        text,
        flags=re.I | re.S,
    )
    if blocks:
        return blocks[-1].strip()
    value = text.strip()
    value = re.sub(r"^\s*FINAL\s+ANSWER\s*:\s*", "", value, flags=re.I)
    return value.strip()


def deterministic_episode_seed(
    base_seed: int,
    generation: int,
    example_key: str,
    replication: int,
) -> int:
    text = f"{base_seed}:{generation}:{example_key}:{replication}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little") % (
        2**32 - 1
    )


def choose_batch(
    data_by_benchmark: Mapping[str, Sequence[BenchmarkExample]],
    batch_per_benchmark: int,
    generation: int,
    seed: int,
) -> list[BenchmarkExample]:
    batch: list[BenchmarkExample] = []
    for position, benchmark in enumerate(sorted(data_by_benchmark)):
        data = data_by_benchmark[benchmark]
        rng = np.random.default_rng(seed + generation * 1009 + position * 9176)
        size = min(batch_per_benchmark, len(data))
        indices = rng.choice(len(data), size=size, replace=False)
        batch.extend(data[int(index)] for index in indices)
    return batch
