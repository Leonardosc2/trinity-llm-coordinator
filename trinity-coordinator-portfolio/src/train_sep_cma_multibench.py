#!/usr/bin/env python3
"""
Sakana/TRINITY-inspired local-pool trainer for multiple benchmarks.

What this file implements:
- Qwen3-0.6B coordinator backbone.
- One linear L+3 head (L model logits + Thinker/Worker/Verifier logits).
- Penultimate generated-token hidden state for coordinator decisions.
- Singular-value scale adaptation for the seven linear matrices in the
  coordinator's second-to-last transformer layer.
- Full transcript at every coordination turn.
- Verifier ACCEPT/REVISE termination, no separate STOP action.
- Balanced MATH-500, MMLU, RLPR, and LiveCodeBench batches.
- Benchmark-specific answer contracts and binary graders.
- Macro-averaged Sep-CMA-ES fitness so large datasets do not dominate.
- Leakage-safe use of official train/test splits where available.
- Optional fully local MMLU, RLPR, and LiveCodeBench dataset files, avoiding
  Hugging Face network access during training.
- Optional warm start from the existing MATH-500 generation-8 checkpoint,
  while deliberately resetting the Sep-CMA-ES optimizer for the new objective.
- Three local OpenAI-compatible vLLM workers on ports 8001, 8002, and 8003.

This is a close practical reproduction, not Sakana AI's private training code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import requests
import sympy as sp
import torch
from cmaes import SepCMA
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    standard_transformations,
    parse_expr,
)
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


ROLES = ("Thinker", "Worker", "Verifier")
SUPPORTED_BENCHMARKS = ("math500", "mmlu", "rlpr", "livecodebench")
TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


@dataclass(frozen=True)
class AgentSpec:
    key: str
    display_name: str
    base_url: str
    served_model_name: str


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


DEFAULT_AGENTS = (
    AgentSpec(
        key="qwen3-4b",
        display_name="Qwen3-4B",
        base_url="http://127.0.0.1:8001/v1",
        served_model_name="qwen3-4b",
    ),
    AgentSpec(
        key="deepseek-r1-qwen-7b",
        display_name="DeepSeek-R1-Distill-Qwen-7B",
        base_url="http://127.0.0.1:8002/v1",
        served_model_name="deepseek-r1-qwen-7b",
    ),
    AgentSpec(
        key="qwen2.5-coder-3b",
        display_name="Qwen2.5-Coder-3B-Instruct",
        base_url="http://127.0.0.1:8003/v1",
        served_model_name="qwen2.5-coder-3b",
    ),
)


ROLE_SYSTEM_PROMPTS = {
    "Thinker": (
        "You are the THINKER in a multi-model team. Analyze the entire problem and transcript. "
        "Develop a concise high-level strategy, identify mistakes or missing subgoals, and guide "
        "the next solver. Do not pretend an unverified answer is correct."
    ),
    "Worker": (
        "You are the WORKER in a multi-model team. Use the entire problem and transcript to make "
        "concrete progress and solve the task. Show enough reasoning to be checked. Follow the "
        "task-specific answer contract included after the transcript."
    ),
    "Verifier": (
        "You are the VERIFIER in a multi-model team. Independently check the accumulated solution "
        "for correctness, completeness, and responsiveness. Your first line must be exactly either "
        "JUDGMENT: ACCEPT or JUDGMENT: REVISE. If ACCEPT, restate the verified result using the "
        "task-specific answer contract. If REVISE, diagnose the error, correct it when possible, "
        "and still provide the best corrected result using that contract. Never output ACCEPT "
        "merely because another agent sounded confident."
    ),
}


class VLLMAgentClient:
    def __init__(
        self,
        specs: Sequence[AgentSpec],
        timeout_seconds: int = 600,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        retries: int = 2,
    ) -> None:
        self.specs = tuple(specs)
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.session = requests.Session()

    def healthcheck(self) -> None:
        for spec in self.specs:
            url = f"{spec.base_url.rstrip('/')}/models"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            names = [item.get("id") for item in payload.get("data", [])]
            if spec.served_model_name not in names:
                raise RuntimeError(
                    f"{spec.display_name} is reachable at {url}, but served name "
                    f"{spec.served_model_name!r} was not found. Available: {names}"
                )
            print(f"[ready] {spec.display_name} -> {spec.base_url}", flush=True)

    def complete(
        self,
        agent_index: int,
        role: str,
        transcript: str,
        answer_contract: str,
    ) -> str:
        spec = self.specs[agent_index]
        url = f"{spec.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": spec.served_model_name,
            "messages": [
                {"role": "system", "content": ROLE_SYSTEM_PROMPTS[role]},
                {
                    "role": "user",
                    "content": (
                        "Below is the original task and the complete collaboration transcript.\n\n"
                        f"{transcript}\n\n"
                        f"TASK-SPECIFIC ANSWER CONTRACT:\n{answer_contract}\n\n"
                        f"Act only as the {role.upper()} now."
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 2):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout_seconds)
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("vLLM returned an empty message")
                return text.strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt <= self.retries:
                    wait = 2**attempt
                    print(
                        f"[retry {attempt}/{self.retries}] {spec.display_name}: {exc}; "
                        f"waiting {wait}s",
                        flush=True,
                    )
                    time.sleep(wait)
        raise RuntimeError(f"Agent call failed for {spec.display_name}: {last_error}")


class SVDScaleAdapter:
    """Keeps U and Vh fixed and changes only log singular-value scales."""

    def __init__(self, name: str, module: nn.Linear, device: torch.device) -> None:
        self.name = name
        self.module = module
        self.device = device
        self.weight_dtype = module.weight.dtype

        print(
            f"[svd] factorizing {name} with shape {tuple(module.weight.shape)} ...",
            flush=True,
        )
        with torch.no_grad():
            weight = module.weight.detach().to(device=device, dtype=torch.float32)
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            # Storing the orthogonal factors in BF16 saves memory. Singular values stay FP32.
            self.u = u.to(dtype=self.weight_dtype)
            self.base_s = s.to(dtype=torch.float32)
            self.vh = vh.to(dtype=self.weight_dtype)
            self.rank = int(s.numel())
            del weight, u, s, vh
            torch.cuda.empty_cache() if device.type == "cuda" else None
        print(f"[svd] {name}: {self.rank} evolvable scales", flush=True)

    @torch.inference_mode()
    def apply(self, log_scales: torch.Tensor) -> None:
        if log_scales.numel() != self.rank:
            raise ValueError(
                f"{self.name}: expected {self.rank} scales, got {log_scales.numel()}"
            )
        # Bounds are also enforced by SepCMA. Clamp again for numerical safety.
        bounded = log_scales.to(self.device, torch.float32).clamp(-0.50, 0.50)
        scaled_s = (self.base_s * torch.exp(bounded)).to(self.weight_dtype)
        reconstructed = (self.u * scaled_s.unsqueeze(0)) @ self.vh
        self.module.weight.copy_(reconstructed.to(self.module.weight.dtype))
        del reconstructed, scaled_s, bounded


class TrinityCoordinator:
    def __init__(
        self,
        model_path: str,
        agents: Sequence[AgentSpec],
        device: str,
        max_input_tokens: int = 4096,
        coordinator_new_tokens: int = 8,
        use_svd_tuning: bool = True,
        seed: int = 0,
    ) -> None:
        self.agents = tuple(agents)
        self.roles = ROLES
        self.device = torch.device(device)
        self.max_input_tokens = max_input_tokens
        self.coordinator_new_tokens = coordinator_new_tokens
        self.use_svd_tuning = use_svd_tuning

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print(f"[coordinator] loading tokenizer from {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        print(f"[coordinator] loading Qwen3-0.6B on {self.device}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        hidden_size = int(self.model.config.hidden_size)
        self.output_size = len(self.agents) + len(self.roles)
        # Sakana describes one linear L+3 head, then splits its output into two groups.
        self.head = nn.Linear(hidden_size, self.output_size, bias=True).to(
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.head.bias)
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)

        self.svd_adapters: list[SVDScaleAdapter] = []
        if use_svd_tuning:
            layer = self._second_to_last_layer()
            targets = (
                ("self_attn.q_proj", layer.self_attn.q_proj),
                ("self_attn.k_proj", layer.self_attn.k_proj),
                ("self_attn.v_proj", layer.self_attn.v_proj),
                ("self_attn.o_proj", layer.self_attn.o_proj),
                ("mlp.gate_proj", layer.mlp.gate_proj),
                ("mlp.up_proj", layer.mlp.up_proj),
                ("mlp.down_proj", layer.mlp.down_proj),
            )
            for name, module in targets:
                if not isinstance(module, nn.Linear):
                    raise TypeError(f"Expected nn.Linear at {name}, got {type(module)}")
                self.svd_adapters.append(SVDScaleAdapter(name, module, self.device))

        self._head_weight_count = self.head.weight.numel()
        self._head_bias_count = self.head.bias.numel()
        self._svd_count = sum(adapter.rank for adapter in self.svd_adapters)
        self.vector_dim = self._head_weight_count + self._head_bias_count + self._svd_count
        print(
            "[coordinator] evolvable parameters: "
            f"head={self._head_weight_count + self._head_bias_count:,}, "
            f"svd={self._svd_count:,}, total={self.vector_dim:,}",
            flush=True,
        )
        if self.vector_dim >= 20_000:
            print(
                "[warning] total evolvable parameters are not below 20K on this model build",
                flush=True,
            )

    def _second_to_last_layer(self) -> nn.Module:
        possible = []
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            possible.append(self.model.model.layers)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            possible.append(self.model.transformer.h)
        if hasattr(self.model, "layers"):
            possible.append(self.model.layers)
        for layers in possible:
            if len(layers) >= 2:
                print(
                    f"[coordinator] applying SVD scales to transformer layer {len(layers) - 2}",
                    flush=True,
                )
                return layers[-2]
        raise RuntimeError("Could not locate the coordinator transformer layers")

    def initial_vector(self) -> np.ndarray:
        pieces = [
            self.head.weight.detach().cpu().reshape(-1).double().numpy(),
            self.head.bias.detach().cpu().reshape(-1).double().numpy(),
        ]
        if self._svd_count:
            pieces.append(np.zeros(self._svd_count, dtype=np.float64))
        return np.concatenate(pieces).astype(np.float64, copy=False)

    def bounds(self) -> np.ndarray:
        # Head values are allowed a wider range; SVD log-scales correspond to exp(-0.5..0.5).
        head_count = self._head_weight_count + self._head_bias_count
        lower = np.concatenate(
            [
                np.full(head_count, -1.0, dtype=np.float64),
                np.full(self._svd_count, -0.5, dtype=np.float64),
            ]
        )
        upper = np.concatenate(
            [
                np.full(head_count, 1.0, dtype=np.float64),
                np.full(self._svd_count, 0.5, dtype=np.float64),
            ]
        )
        return np.stack([lower, upper], axis=1)

    @torch.inference_mode()
    def set_vector(self, vector: np.ndarray) -> None:
        if vector.shape != (self.vector_dim,):
            raise ValueError(f"Expected vector shape {(self.vector_dim,)}, got {vector.shape}")
        cursor = 0
        w_end = cursor + self._head_weight_count
        weight = torch.from_numpy(vector[cursor:w_end]).to(
            self.device, dtype=self.head.weight.dtype
        )
        self.head.weight.copy_(weight.view_as(self.head.weight))
        cursor = w_end

        b_end = cursor + self._head_bias_count
        bias = torch.from_numpy(vector[cursor:b_end]).to(
            self.device, dtype=self.head.bias.dtype
        )
        self.head.bias.copy_(bias.view_as(self.head.bias))
        cursor = b_end

        for adapter in self.svd_adapters:
            end = cursor + adapter.rank
            scales = torch.from_numpy(vector[cursor:end])
            adapter.apply(scales)
            cursor = end
        if cursor != self.vector_dim:
            raise RuntimeError("Parameter-vector cursor mismatch")

    def _chat_prompt(self, state_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a small routing coordinator. Read the complete task transcript. "
                    "Your generated text will be discarded; its hidden representation is used "
                    "to select the next model and role."
                ),
            },
            {"role": "user", "content": state_text},
        ]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    @torch.inference_mode()
    def logits(self, state_text: str) -> tuple[np.ndarray, np.ndarray]:
        prompt = self._chat_prompt(state_text)
        reserve = max(2, self.coordinator_new_tokens)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(32, self.max_input_tokens - reserve),
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        generated = self.model.generate(
            **encoded,
            max_new_tokens=self.coordinator_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Re-run the short completed sequence so the penultimate output token's final hidden
        # state is obtained unambiguously. Coordinator text itself is never shown to agents.
        attention_mask = torch.ones_like(generated, device=self.device)
        outputs = self.model(
            input_ids=generated,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        token_index = -2 if generated.shape[1] >= 2 else -1
        hidden = outputs.hidden_states[-1][:, token_index, :].to(torch.float32)
        combined = self.head(hidden).squeeze(0)
        model_logits = combined[: len(self.agents)].detach().cpu().numpy()
        role_logits = combined[len(self.agents) :].detach().cpu().numpy()
        return model_logits, role_logits

    def choose_action(
        self,
        state_text: str,
        rng: np.random.Generator,
    ) -> tuple[int, int, dict[str, list[float]]]:
        model_logits, role_logits = self.logits(state_text)
        model_probs = stable_softmax(model_logits)
        role_probs = stable_softmax(role_logits)
        model_index = categorical_from_uniform(model_probs, float(rng.random()))
        role_index = categorical_from_uniform(role_probs, float(rng.random()))
        return model_index, role_index, {
            "model_probs": model_probs.tolist(),
            "role_probs": role_probs.tolist(),
        }

    def export_candidate(self, vector: np.ndarray, path: Path, metadata: dict[str, Any]) -> None:
        head_end = self._head_weight_count + self._head_bias_count
        payload = {
            "format": "trinity-sep-cma-v1",
            "agents": [asdict(spec) for spec in self.agents],
            "roles": list(self.roles),
            "vector_dim": self.vector_dim,
            "head_parameter_count": head_end,
            "svd_parameter_count": self._svd_count,
            "vector": torch.from_numpy(vector.copy()),
            "metadata": metadata,
        }
        torch.save(payload, path)


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
    coordinator: TrinityCoordinator,
    client: VLLMAgentClient,
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


def extract_gold(record: dict[str, Any]) -> str:
    for key in ("answer", "final_answer", "target", "gold", "label"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    solution = str(record.get("solution", ""))
    boxed = extract_last_boxed(solution)
    if boxed:
        return boxed
    return extract_final_answer(solution)


def load_excluded_math_indices(path: Path) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_indices = payload
    elif isinstance(payload, dict):
        raw_indices = next(
            (
                payload[key]
                for key in ("indices", "holdout_indices", "selected_indices")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
        if raw_indices is None:
            raise ValueError(
                f"{path} must contain a list under 'indices', 'holdout_indices', "
                "or 'selected_indices'"
            )
    else:
        raise ValueError(f"{path} must contain a JSON list or object")
    return {int(index) for index in raw_indices}


def load_math500(
    path: Path,
    limit: Optional[int] = None,
    excluded_indices: Optional[set[int]] = None,
) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    excluded = excluded_indices or set()
    usable_index = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            question = next(
                (
                    str(record[key]).strip()
                    for key in ("problem", "question", "prompt", "input")
                    if record.get(key) is not None and str(record[key]).strip()
                ),
                "",
            )
            gold = extract_gold(record)
            if question and gold:
                # Holdout indices are positions in the evaluator's usable-example
                # list, not raw JSONL line numbers.
                source_index = usable_index
                usable_index += 1
                if source_index in excluded:
                    continue
                examples.append(
                    BenchmarkExample(
                        benchmark="math500",
                        uid=str(source_index),
                        question=question,
                        gold=gold,
                        grader="math",
                        answer_contract=(
                            "End with exactly one line: FINAL ANSWER: <answer>."
                        ),
                        metadata={"source_index": source_index},
                    )
                )
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise RuntimeError(f"No usable examples were loaded from {path}")
    print(
        f"[data] loaded {len(examples)} MATH examples from {path}; "
        f"excluded {len(excluded)} protected holdout indices",
        flush=True,
    )
    return examples


def import_hf_datasets() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face datasets is required. Install it with: "
            "python3 -m pip install datasets"
        ) from exc
    return load_dataset


def iter_local_records(path: Path) -> Any:
    """Yield records from a local Parquet, JSONL, or JSON dataset file."""
    if not path.is_file():
        raise FileNotFoundError(f"Local dataset file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Reading local Parquet files requires pyarrow. Install it with: "
                "python3 -m pip install pyarrow"
            ) from exc
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=2048):
            yield from batch.to_pylist()
        return

    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise ValueError(
                        f"Expected a JSON object at {path}:{line_number}"
                    )
                yield record
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            records = payload.get("data")
        else:
            records = payload
        if not isinstance(records, list):
            raise ValueError(
                f"Expected a JSON list or an object with a 'data' list in {path}"
            )
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"Expected every record in {path} to be an object")
            yield record
        return

    raise ValueError(
        f"Unsupported local dataset format for {path}; use Parquet, JSONL, or JSON"
    )


def local_file_manifest(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Local dataset file does not exist: {resolved}")
    return {
        "local_file": str(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def load_mmlu(
    source: str,
    config: str,
    split: str,
    limit: Optional[int],
    local_file: Optional[Path] = None,
) -> tuple[list[BenchmarkExample], dict[str, Any]]:
    if local_file is not None:
        resolved_local_file = local_file.expanduser().resolve()
        dataset = iter_local_records(resolved_local_file)
        source_description = str(resolved_local_file)
    else:
        load_dataset = import_hf_datasets()
        dataset = load_dataset(source, config, split=split)
        source_description = f"{source}/{config}:{split}"

    examples: list[BenchmarkExample] = []
    letters = "ABCD"
    for source_index, record in enumerate(dataset):
        question = str(record.get("question", "")).strip()
        choices = [str(item).strip() for item in record.get("choices", [])]
        try:
            answer_index = int(record.get("answer"))
        except (TypeError, ValueError):
            continue
        if not question or len(choices) != 4 or answer_index not in range(4):
            continue
        rendered_choices = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(letters, choices)
        )
        examples.append(
            BenchmarkExample(
                benchmark="mmlu",
                uid=str(source_index),
                question=f"{question}\n\n{rendered_choices}",
                gold=letters[answer_index],
                grader="choice",
                answer_contract=(
                    "Choose one option. End with exactly one line: FINAL ANSWER: "
                    "A, B, C, or D."
                ),
                metadata={
                    "source_index": source_index,
                    "subject": str(record.get("subject", "")),
                },
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise RuntimeError(f"No usable MMLU examples in {source_description}")
    if local_file is not None:
        manifest = {
            **local_file_manifest(resolved_local_file),
            "split": split,
            "count": len(examples),
        }
    else:
        manifest = {
            "source": source,
            "config": config,
            "split": split,
            "fingerprint": str(getattr(dataset, "_fingerprint", "")),
            "count": len(examples),
        }
    print(
        f"[data] loaded {len(examples)} MMLU examples from {source_description}",
        flush=True,
    )
    return examples, manifest


def rlpr_prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, list):
        parts: list[str] = []
        for message in prompt:
            if isinstance(message, Mapping):
                role = str(message.get("role", "user")).upper()
                content = str(message.get("content", "")).strip()
                if content:
                    parts.append(f"{role}: {content}")
            elif str(message).strip():
                parts.append(str(message).strip())
        return "\n\n".join(parts)
    return str(prompt).strip()


def load_rlpr(
    source: str,
    split: str,
    limit: Optional[int],
    local_file: Optional[Path] = None,
) -> tuple[list[BenchmarkExample], dict[str, Any]]:
    if local_file is not None:
        resolved_local_file = local_file.expanduser().resolve()
        dataset = iter_local_records(resolved_local_file)
        source_description = str(resolved_local_file)
    else:
        load_dataset = import_hf_datasets()
        dataset = load_dataset(source, split=split)
        source_description = f"{source}:{split}"

    examples: list[BenchmarkExample] = []
    for source_index, record in enumerate(dataset):
        question = rlpr_prompt_to_text(record.get("prompt", ""))
        reward_model = record.get("reward_model") or {}
        if not isinstance(reward_model, Mapping):
            continue
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, (dict, list)):
            gold = json.dumps(ground_truth, ensure_ascii=False, sort_keys=True)
        else:
            gold = str(ground_truth or "").strip()
        if not question or not gold:
            continue
        extra_info = record.get("extra_info") or {}
        if not isinstance(extra_info, Mapping):
            extra_info = {}
        external_id = str(extra_info.get("id", source_index))
        uid = f"{source_index}:{external_id}"
        examples.append(
            BenchmarkExample(
                benchmark="rlpr",
                uid=uid,
                question=question,
                gold=gold,
                grader="rlpr_rule",
                answer_contract=(
                    "Give the requested result. End with exactly one line: "
                    "FINAL ANSWER: <answer>."
                ),
                metadata={
                    "source_index": source_index,
                    "ability": str(record.get("ability", "")),
                    "category": str(extra_info.get("category", "")),
                    "answer_type": str(extra_info.get("answer_type", "")),
                    "reward_style": str(reward_model.get("style", "")),
                },
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise RuntimeError(f"No usable RLPR examples in {source_description}")
    if local_file is not None:
        manifest = {
            **local_file_manifest(resolved_local_file),
            "split": split,
            "count": len(examples),
        }
    else:
        manifest = {
            "source": source,
            "split": split,
            "fingerprint": str(getattr(dataset, "_fingerprint", "")),
            "count": len(examples),
        }
    print(
        f"[data] loaded {len(examples)} RLPR examples from {source_description}",
        flush=True,
    )
    return examples, manifest


class LiveCodeBenchGrader:
    """Thin wrapper around LiveCodeBench's official execution-based evaluator."""

    def __init__(self, samples: Mapping[str, dict[str, Any]], timeout: int) -> None:
        try:
            from lcb_runner.evaluation.compute_code_generation_metrics import (
                check_correctness,
            )
        except ImportError as exc:
            raise RuntimeError(
                "LiveCodeBench is not installed. Clone its official repository and run "
                "`python3 -m pip install -e .` inside that repository."
            ) from exc
        self.samples = dict(samples)
        self.timeout = timeout
        self._check_correctness = check_correctness

    def score(self, example: BenchmarkExample, response: str) -> tuple[int, str, Any]:
        code = extract_python_solution(response)
        if not code:
            return 0, "", {"reason": "no_code"}
        result = self._check_correctness(
            self.samples[example.uid],
            code,
            timeout=self.timeout,
            debug=False,
        )
        if isinstance(result, tuple):
            test_results, metadata = result
        else:
            test_results, metadata = result, None
        if isinstance(test_results, (list, tuple)):
            passed = bool(test_results) and all(
                item is True or item == 1 for item in test_results
            )
        else:
            passed = test_results is True or test_results == 1
        details = {
            "test_results": make_json_safe(test_results),
            "metadata": make_json_safe(metadata),
        }
        return int(passed), code, details


def make_json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def load_livecodebench(
    release: str,
    limit: Optional[int],
    timeout: int,
    local_file: Optional[Path] = None,
) -> tuple[list[BenchmarkExample], dict[str, Any], LiveCodeBenchGrader]:
    try:
        from lcb_runner.benchmarks.code_generation import (
            CodeGenerationProblem,
            load_code_generation_dataset,
        )
    except ImportError as exc:
        raise RuntimeError(
            "LiveCodeBench is not installed. Clone its official repository and run "
            "`python3 -m pip install -e .` inside that repository."
        ) from exc

    if local_file is not None:
        if release not in {"v1", "release_v1"}:
            raise ValueError(
                "The local LiveCodeBench V1 test.jsonl file must be used with "
                "--lcb_train_release v1"
            )
        resolved_local_file = local_file.expanduser().resolve()
        problems = (
            CodeGenerationProblem(**dict(record))
            for record in iter_local_records(resolved_local_file)
        )
        source_description = str(resolved_local_file)
    else:
        try:
            problems = load_code_generation_dataset(release_version=release)
        except TypeError:
            problems = load_code_generation_dataset(release)
        source_description = f"livecodebench/code_generation_lite:{release}"

    examples: list[BenchmarkExample] = []
    samples: dict[str, dict[str, Any]] = {}
    for source_index, problem in enumerate(problems):
        external_id = str(
            getattr(problem, "question_id", None)
            or getattr(problem, "contest_id", None)
            or source_index
        )
        uid = f"{source_index}:{external_id}"
        question = str(
            getattr(problem, "question_content", None)
            or getattr(problem, "question", "")
        ).strip()
        starter_code = str(getattr(problem, "starter_code", "") or "").strip()
        if starter_code:
            question += f"\n\nSTARTER CODE:\n```python\n{starter_code}\n```"
        if not question:
            continue
        sample = problem.get_evaluation_sample()
        examples.append(
            BenchmarkExample(
                benchmark="livecodebench",
                uid=uid,
                question=question,
                gold="<hidden tests>",
                grader="livecodebench",
                answer_contract=(
                    "Return the complete Python 3 solution in one fenced ```python "
                    "code block. Do not add a FINAL ANSWER line."
                ),
                metadata={"source_index": source_index, "release": release},
            )
        )
        samples[uid] = sample
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise RuntimeError(f"No usable LiveCodeBench examples in {source_description}")
    if local_file is not None:
        manifest = {
            **local_file_manifest(resolved_local_file),
            "release": release,
            "count": len(examples),
        }
    else:
        manifest = {
            "source": "livecodebench/code_generation_lite",
            "release": release,
            "count": len(examples),
        }
    print(
        f"[data] loaded {len(examples)} LiveCodeBench examples from "
        f"{source_description}",
        flush=True,
    )
    return examples, manifest, LiveCodeBenchGrader(samples, timeout)


def strip_outer_math(text: str) -> str:
    value = text.strip()
    for left, right in (("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if value.startswith(left) and value.endswith(right):
            value = value[len(left) : -len(right)].strip()
    return value


def replace_latex_frac(text: str) -> str:
    pattern = re.compile(r"\\(?:d?frac)\s*\{")
    cursor = 0
    output = ""
    while True:
        match = pattern.search(text, cursor)
        if not match:
            output += text[cursor:]
            break
        output += text[cursor : match.start()]
        first_open = text.find("{", match.start())
        numerator = extract_balanced_braces(text, first_open)
        if numerator is None:
            output += text[match.start() : match.end()]
            cursor = match.end()
            continue
        first_close = first_open + len(numerator) + 1
        second_open = text.find("{", first_close + 1)
        if second_open < 0:
            output += text[match.start() : match.end()]
            cursor = match.end()
            continue
        denominator = extract_balanced_braces(text, second_open)
        if denominator is None:
            output += text[match.start() : match.end()]
            cursor = match.end()
            continue
        second_close = second_open + len(denominator) + 1
        output += f"(({replace_latex_frac(numerator)})/({replace_latex_frac(denominator)}))"
        cursor = second_close + 1
    return output


def replace_latex_sqrt(text: str) -> str:
    pattern = re.compile(r"\\sqrt\s*\{")
    cursor = 0
    output = ""
    while True:
        match = pattern.search(text, cursor)
        if not match:
            output += text[cursor:]
            break
        output += text[cursor : match.start()]
        brace = text.find("{", match.start())
        inside = extract_balanced_braces(text, brace)
        if inside is None:
            output += text[match.start() : match.end()]
            cursor = match.end()
            continue
        close = brace + len(inside) + 1
        output += f"sqrt({replace_latex_sqrt(inside)})"
        cursor = close + 1
    return output


def normalize_answer_string(text: str) -> str:
    value = strip_outer_math(text)
    value = value.replace("−", "-").replace("–", "-")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\!", "")
    value = value.replace("\\cdot", "*").replace("\\times", "*")
    value = value.replace("\\pi", "pi")
    value = value.replace("∞", "oo").replace("\\infty", "oo")
    value = replace_latex_frac(value)
    value = replace_latex_sqrt(value)
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace("{", "(").replace("}", ")")
    value = value.replace("^", "**")
    value = value.replace("%", "/100")
    value = value.strip().rstrip(". ")
    # For simple assignments, compare the value on the right.
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\s*=.+", value):
        value = value.split("=", 1)[1]
    value = re.sub(r"\s+", "", value)
    return value


def split_top_level(value: str) -> list[str]:
    if value.startswith(("[", "(", "{")) and value.endswith(("]", ")", "}")):
        value = value[1:-1]
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in value:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return [part for part in parts if part]


def to_sympy(value: str) -> Optional[sp.Expr]:
    try:
        local_dict = {
            "pi": sp.pi,
            "e": sp.E,
            "E": sp.E,
            "oo": sp.oo,
            "sqrt": sp.sqrt,
        }
        return parse_expr(
            value,
            local_dict=local_dict,
            transformations=TRANSFORMS,
            evaluate=True,
        )
    except Exception:  # noqa: BLE001
        return None


def answers_equivalent(predicted: str, gold: str) -> bool:
    pred = normalize_answer_string(predicted)
    target = normalize_answer_string(gold)
    if not pred or not target:
        return False
    if pred.lower() == target.lower():
        return True

    pred_parts = split_top_level(pred)
    target_parts = split_top_level(target)
    if len(pred_parts) == len(target_parts) and len(pred_parts) > 1:
        unmatched = target_parts.copy()
        for item in pred_parts:
            match_index = next(
                (
                    i
                    for i, target_item in enumerate(unmatched)
                    if answers_equivalent(item, target_item)
                ),
                None,
            )
            if match_index is None:
                return False
            unmatched.pop(match_index)
        return not unmatched

    pred_expr = to_sympy(pred)
    target_expr = to_sympy(target)
    if pred_expr is None or target_expr is None:
        return False
    try:
        difference = sp.simplify(pred_expr - target_expr)
        if difference == 0:
            return True
        numeric = complex(sp.N(difference, 30))
        return abs(numeric) <= 1e-9
    except Exception:  # noqa: BLE001
        return False


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


def normalized_text_answer(text: str) -> str:
    value = clean_extracted_answer(text).casefold()
    value = value.replace("−", "-").replace("–", "-")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n.,;:!?\"'")


def rlpr_rule_score(predicted: str, gold: str) -> bool:
    if answers_equivalent(predicted, gold):
        return True
    pred = normalized_text_answer(predicted)
    target = normalized_text_answer(gold)
    if pred and pred == target:
        return True
    # Some RLPR ground truths are lists of acceptable answers.
    try:
        decoded = json.loads(gold)
    except (json.JSONDecodeError, TypeError):
        decoded = None
    if isinstance(decoded, list):
        return any(
            rlpr_rule_score(predicted, str(item))
            for item in decoded
            if not isinstance(item, (dict, list))
        )
    if isinstance(decoded, dict):
        for key in ("answer", "answers", "ground_truth", "target"):
            if key in decoded:
                value = decoded[key]
                values = value if isinstance(value, list) else [value]
                return any(rlpr_rule_score(predicted, str(item)) for item in values)
    return False


def score_example(
    example: BenchmarkExample,
    episode: Mapping[str, Any],
    lcb_grader: Optional[LiveCodeBenchGrader],
) -> tuple[int, str, Any]:
    final_text = str(episode.get("final_text", ""))
    if example.grader == "math":
        predicted = extract_final_answer(final_text)
        return int(answers_equivalent(predicted, example.gold)), predicted, None
    if example.grader == "choice":
        predicted = extract_choice_answer(final_text)
        return int(predicted == example.gold.upper()), predicted, None
    if example.grader == "rlpr_rule":
        predicted = extract_final_answer(final_text)
        return int(rlpr_rule_score(predicted, example.gold)), predicted, {
            "mode": "rule_based"
        }
    if example.grader == "livecodebench":
        if lcb_grader is None:
            raise RuntimeError("LiveCodeBench grader was not initialized")
        return lcb_grader.score(example, final_text)
    raise ValueError(f"Unsupported grader: {example.grader}")


def evaluate_candidate(
    coordinator: TrinityCoordinator,
    client: VLLMAgentClient,
    vector: np.ndarray,
    batch: Sequence[BenchmarkExample],
    generation: int,
    base_seed: int,
    max_turns: int,
    replications: int,
    candidate_number: int,
    log_handle: Any,
    verbose_episodes: bool,
    lcb_grader: Optional[LiveCodeBenchGrader],
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    coordinator.set_vector(vector)
    rewards_by_benchmark: dict[str, list[int]] = defaultdict(list)
    records: list[dict[str, Any]] = []

    for example in batch:
        for replication in range(replications):
            seed = deterministic_episode_seed(
                base_seed, generation, example.key, replication
            )
            started = time.time()
            try:
                episode = run_episode(
                    coordinator=coordinator,
                    client=client,
                    question=example.question,
                    answer_contract=example.answer_contract,
                    max_turns=max_turns,
                    seed=seed,
                    verbose=verbose_episodes,
                )
                reward, predicted, grader_details = score_example(
                    example,
                    episode,
                    lcb_grader,
                )
                error = None
            except Exception as exc:  # noqa: BLE001
                episode = {
                    "turns": [],
                    "accepted": False,
                    "final_text": "",
                    "predicted_answer": "",
                }
                predicted = ""
                grader_details = None
                reward = 0
                error = f"{type(exc).__name__}: {exc}"
                print(
                    f"      [episode failure] example={example.key}: {error}",
                    flush=True,
                )
            rewards_by_benchmark[example.benchmark].append(reward)
            record = {
                "generation": generation,
                "candidate": candidate_number,
                "benchmark": example.benchmark,
                "example_uid": example.uid,
                "example_key": example.key,
                "grader": example.grader,
                "replication": replication,
                "reward": reward,
                "gold": example.gold,
                "predicted": predicted,
                "final_text": episode["final_text"],
                "grader_details": make_json_safe(grader_details),
                "accepted": episode["accepted"],
                "turns": episode["turns"],
                "seconds": round(time.time() - started, 3),
                "error": error,
            }
            records.append(record)
            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_handle.flush()

    per_benchmark = {
        benchmark: float(np.mean(rewards))
        for benchmark, rewards in rewards_by_benchmark.items()
    }
    macro_score = float(np.mean(list(per_benchmark.values()))) if per_benchmark else 0.0
    return macro_score, per_benchmark, records


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


def parse_benchmark_names(value: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip()))
    unknown = [name for name in names if name not in SUPPORTED_BENCHMARKS]
    if unknown:
        raise ValueError(
            f"Unknown benchmarks {unknown}; choose from {list(SUPPORTED_BENCHMARKS)}"
        )
    if not names:
        raise ValueError("--benchmarks cannot be empty")
    return names


def optional_limit(value: int) -> Optional[int]:
    return value if value > 0 else None


def load_training_benchmarks(
    args: argparse.Namespace,
    benchmark_names: Sequence[str],
) -> tuple[
    dict[str, list[BenchmarkExample]],
    dict[str, dict[str, Any]],
    Optional[LiveCodeBenchGrader],
]:
    data: dict[str, list[BenchmarkExample]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    lcb_grader: Optional[LiveCodeBenchGrader] = None

    if "math500" in benchmark_names:
        if not args.math_exclude_indices:
            raise ValueError(
                "MATH-500 training requires --math_exclude_indices pointing to the "
                "new holdout JSON. This prevents evaluation questions from leaking "
                "back into training."
            )
        math_path = Path(args.math_data).expanduser().resolve()
        exclusion_path = Path(args.math_exclude_indices).expanduser().resolve()
        excluded = load_excluded_math_indices(exclusion_path)
        data["math500"] = load_math500(
            math_path,
            limit=optional_limit(args.math_limit),
            excluded_indices=excluded,
        )
        manifests["math500"] = {
            "source": str(math_path),
            "split": "local_train_minus_holdout",
            "excluded_indices_file": str(exclusion_path),
            "excluded_count": len(excluded),
            "count": len(data["math500"]),
        }

    if "mmlu" in benchmark_names:
        data["mmlu"], manifests["mmlu"] = load_mmlu(
            args.mmlu_source,
            args.mmlu_config,
            args.mmlu_train_split,
            optional_limit(args.mmlu_limit),
            (
                Path(args.mmlu_local_file).expanduser().resolve()
                if args.mmlu_local_file
                else None
            ),
        )

    if "rlpr" in benchmark_names:
        data["rlpr"], manifests["rlpr"] = load_rlpr(
            args.rlpr_source,
            args.rlpr_train_split,
            optional_limit(args.rlpr_limit),
            (
                Path(args.rlpr_local_file).expanduser().resolve()
                if args.rlpr_local_file
                else None
            ),
        )

    if "livecodebench" in benchmark_names:
        if not args.allow_code_execution:
            raise ValueError(
                "LiveCodeBench executes generated Python. Run in an isolated container "
                "and pass --allow_code_execution to confirm that setup."
            )
        (
            data["livecodebench"],
            manifests["livecodebench"],
            lcb_grader,
        ) = load_livecodebench(
            args.lcb_train_release,
            optional_limit(args.lcb_limit),
            args.lcb_timeout,
            (
                Path(args.lcb_local_file).expanduser().resolve()
                if args.lcb_local_file
                else None
            ),
        )

    return data, manifests, lcb_grader


def make_run_signature(
    args: argparse.Namespace,
    manifests: Mapping[str, Mapping[str, Any]],
) -> str:
    payload = {
        "manifests": manifests,
        "coordinator_model": str(Path(args.coordinator_model).expanduser()),
        "use_svd": not args.no_svd,
        "batch_per_benchmark": args.batch_per_benchmark,
        "replications": args.replications,
        "max_turns": args.max_turns,
        "agents": [asdict(spec) for spec in DEFAULT_AGENTS],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_warm_start_vector(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    vector = payload.get("vector") if isinstance(payload, Mapping) else payload
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().cpu().numpy()
    return np.asarray(vector, dtype=np.float64)


def save_pickle_atomic(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TRINITY-inspired balanced multibench Sep-CMA-ES trainer for a local "
            "three-model pool"
        )
    )
    parser.add_argument(
        "--coordinator_model",
        default=str(Path.home() / "tingmin_trinity/models/Qwen3-0.6B"),
    )
    parser.add_argument(
        "--benchmarks",
        default="math500,mmlu,rlpr,livecodebench",
        help="Comma-separated subset of: math500,mmlu,rlpr,livecodebench",
    )
    parser.add_argument(
        "--math_data",
        default=str(Path.home() / "tingmin_trinity/data/math500_ready.jsonl"),
    )
    parser.add_argument(
        "--math_exclude_indices",
        default="",
        help="Required for MATH-500: JSON file containing the protected holdout indices",
    )
    parser.add_argument("--math_limit", type=int, default=0)
    parser.add_argument("--mmlu_source", default="cais/mmlu")
    parser.add_argument("--mmlu_config", default="all")
    parser.add_argument("--mmlu_train_split", default="auxiliary_train")
    parser.add_argument(
        "--mmlu_local_file",
        default="",
        help=(
            "Optional local MMLU auxiliary_train Parquet/JSONL file. When set, "
            "--mmlu_source is not downloaded."
        ),
    )
    parser.add_argument("--mmlu_limit", type=int, default=0)
    parser.add_argument("--rlpr_source", default="openbmb/RLPR-Train-Dataset")
    parser.add_argument("--rlpr_train_split", default="train")
    parser.add_argument(
        "--rlpr_local_file",
        default="",
        help=(
            "Optional local RLPR train Parquet/JSONL file. When set, --rlpr_source "
            "is not downloaded."
        ),
    )
    parser.add_argument("--rlpr_limit", type=int, default=0)
    parser.add_argument(
        "--lcb_train_release",
        default="v1",
        help="TRINITY used the 400-problem LiveCodeBench V1 group for training",
    )
    parser.add_argument(
        "--lcb_local_file",
        default="",
        help=(
            "Optional local LiveCodeBench V1 test.jsonl file. When set, the "
            "LiveCodeBench dataset is not downloaded."
        ),
    )
    parser.add_argument("--lcb_limit", type=int, default=0)
    parser.add_argument("--lcb_timeout", type=int, default=6)
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Required for LiveCodeBench; use only inside an isolated container",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path.home() / "tingmin_trinity/checkpoints/sep_cma_multibench"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument(
        "--population_size",
        type=int,
        default=0,
        help="0 uses ceil(4 + 3 ln(n)), matching the paper's formula",
    )
    parser.add_argument(
        "--batch_per_benchmark",
        type=int,
        default=4,
        help="4 across all four tasks gives 16 examples per candidate",
    )
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent_max_tokens", type=int, default=1024)
    parser.add_argument("--agent_timeout", type=int, default=600)
    parser.add_argument("--coordinator_max_input_tokens", type=int, default=4096)
    parser.add_argument("--coordinator_new_tokens", type=int, default=8)
    parser.add_argument("--no_svd", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--warm_start_checkpoint",
        default="",
        help="Existing best_vector.npy or best_coordinator.pt; resets CMA state",
    )
    parser.add_argument("--verbose_episodes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_names = parse_benchmark_names(args.benchmarks)
    if args.generations <= 0:
        raise ValueError("--generations must be positive")
    if args.batch_per_benchmark <= 0 or args.replications <= 0:
        raise ValueError("--batch_per_benchmark and --replications must be positive")
    if args.max_turns <= 0 or args.lcb_timeout <= 0:
        raise ValueError("--max_turns and --lcb_timeout must be positive")
    if args.resume and args.warm_start_checkpoint:
        raise ValueError("--resume and --warm_start_checkpoint are mutually exclusive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "optimizer_state.pkl"
    log_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "training_summary.json"
    manifest_path = output_dir / "data_manifest.json"
    best_vector_path = output_dir / "best_vector.npy"
    best_candidate_path = output_dir / "best_coordinator.pt"
    existing_artifacts = [
        path
        for path in (
            state_path,
            log_path,
            summary_path,
            manifest_path,
            best_vector_path,
            best_candidate_path,
        )
        if path.exists()
    ]
    if existing_artifacts and not args.resume:
        raise FileExistsError(
            f"{output_dir} already contains training artifacts. Use --resume to "
            "continue it, or choose a new --output_dir to start from generation 0. "
            f"Found: {[path.name for path in existing_artifacts]}"
        )

    print("=" * 78)
    print("TRINITY-STYLE BALANCED MULTIBENCH SEP-CMA-ES TRAINING")
    print("=" * 78)
    print(json.dumps(vars(args), indent=2), flush=True)

    data_by_benchmark, manifests, lcb_grader = load_training_benchmarks(
        args,
        benchmark_names,
    )
    run_signature = make_run_signature(args, manifests)
    manifest_path.write_text(
        json.dumps(
            {
                "benchmarks": list(benchmark_names),
                "run_signature": run_signature,
                "datasets": manifests,
                "held_out_from_training": [
                    "MMLU test",
                    "RLPR Evaluation Dataset",
                    "LiveCodeBench v6",
                    "AIME 2025",
                    "BigCodeBench",
                    "MT-Bench",
                    "GPQA-D",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    coordinator = TrinityCoordinator(
        model_path=str(Path(args.coordinator_model).expanduser()),
        agents=DEFAULT_AGENTS,
        device=args.device,
        max_input_tokens=args.coordinator_max_input_tokens,
        coordinator_new_tokens=args.coordinator_new_tokens,
        use_svd_tuning=not args.no_svd,
        seed=args.seed,
    )

    client = VLLMAgentClient(
        DEFAULT_AGENTS,
        timeout_seconds=args.agent_timeout,
        max_tokens=args.agent_max_tokens,
        temperature=0.0,
        retries=2,
    )
    client.healthcheck()

    auto_population = int(math.ceil(4.0 + 3.0 * math.log(coordinator.vector_dim)))
    population_size = args.population_size if args.population_size > 0 else auto_population
    print(
        f"[sep-cma] dimension={coordinator.vector_dim:,}, "
        f"auto population={auto_population}, using population={population_size}",
        flush=True,
    )

    start_generation = 0
    best_score = -1.0
    best_vector: Optional[np.ndarray] = None
    history: list[dict[str, Any]] = []

    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"Resume requested but state does not exist: {state_path}")
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        optimizer = state["optimizer"]
        start_generation = int(state["next_generation"])
        best_score = float(state["best_score"])
        best_vector = state["best_vector"]
        history = state.get("history", [])
        saved_signature = state.get("run_signature")
        if saved_signature != run_signature:
            raise RuntimeError(
                "Resume configuration/data does not match this checkpoint. Use the "
                "original arguments, or start a new output directory."
            )
        if optimizer.population_size != population_size:
            print(
                f"[resume] checkpoint population {optimizer.population_size} overrides "
                f"requested {population_size}",
                flush=True,
            )
            population_size = optimizer.population_size
        print(f"[resume] continuing from generation {start_generation}", flush=True)
    else:
        if args.warm_start_checkpoint:
            warm_path = Path(args.warm_start_checkpoint).expanduser().resolve()
            if not warm_path.exists():
                raise FileNotFoundError(f"Warm-start checkpoint not found: {warm_path}")
            mean = load_warm_start_vector(warm_path)
            if mean.shape != (coordinator.vector_dim,):
                raise ValueError(
                    f"Warm-start vector shape {mean.shape} does not match coordinator "
                    f"dimension {(coordinator.vector_dim,)}"
                )
            print(
                f"[warm start] mean loaded from {warm_path}; Sep-CMA state reset",
                flush=True,
            )
        else:
            mean = coordinator.initial_vector()
        optimizer = SepCMA(
            mean=mean,
            sigma=args.sigma,
            bounds=coordinator.bounds(),
            population_size=population_size,
            seed=args.seed,
        )

    with log_path.open("a", encoding="utf-8") as log_handle:
        for generation in range(start_generation, start_generation + args.generations):
            generation_started = time.time()
            batch = choose_batch(
                data_by_benchmark,
                args.batch_per_benchmark,
                generation,
                args.seed,
            )
            batch_keys = [example.key for example in batch]
            batch_counts: dict[str, int] = defaultdict(int)
            for example in batch:
                batch_counts[example.benchmark] += 1
            print("\n" + "-" * 78)
            print(
                f"GENERATION {generation} | batch={dict(batch_counts)} | "
                f"population={optimizer.population_size}",
                flush=True,
            )
            solutions: list[tuple[np.ndarray, float]] = []
            generation_scores: list[float] = []
            generation_breakdowns: list[dict[str, float]] = []

            for candidate_number in range(1, optimizer.population_size + 1):
                candidate_started = time.time()
                vector = optimizer.ask()
                print(
                    f"  Candidate {candidate_number}/{optimizer.population_size}",
                    flush=True,
                )
                score, per_benchmark, _records = evaluate_candidate(
                    coordinator=coordinator,
                    client=client,
                    vector=vector,
                    batch=batch,
                    generation=generation,
                    base_seed=args.seed,
                    max_turns=args.max_turns,
                    replications=args.replications,
                    candidate_number=candidate_number,
                    log_handle=log_handle,
                    verbose_episodes=args.verbose_episodes,
                    lcb_grader=lcb_grader,
                )
                fitness = -score  # cmaes minimizes.
                solutions.append((vector, fitness))
                generation_scores.append(score)
                generation_breakdowns.append(per_benchmark)
                breakdown_text = ", ".join(
                    f"{name}={value * 100:.1f}%"
                    for name, value in sorted(per_benchmark.items())
                )
                print(
                    f"    macro={score:.4f} ({score * 100:.1f}%) | "
                    f"{breakdown_text} | "
                    f"time={time.time() - candidate_started:.1f}s",
                    flush=True,
                )

                if score > best_score:
                    best_score = score
                    best_vector = vector.copy()
                    np.save(best_vector_path, best_vector)
                    coordinator.export_candidate(
                        best_vector,
                        best_candidate_path,
                        metadata={
                            "best_score": best_score,
                            "generation": generation,
                            "candidate": candidate_number,
                            "per_benchmark": per_benchmark,
                            "run_signature": run_signature,
                            "args": vars(args),
                        },
                    )
                    print(f"    [new overall best] {best_score:.4f}", flush=True)

            optimizer.tell(solutions)
            generation_best = float(max(generation_scores))
            generation_mean = float(np.mean(generation_scores))
            mean_by_benchmark = {
                benchmark: float(
                    np.mean(
                        [
                            breakdown[benchmark]
                            for breakdown in generation_breakdowns
                            if benchmark in breakdown
                        ]
                    )
                )
                for benchmark in benchmark_names
            }
            generation_record = {
                "generation": generation,
                "batch_keys": batch_keys,
                "batch_counts": dict(batch_counts),
                "population_size": optimizer.population_size,
                "macro_best": generation_best,
                "macro_mean": generation_mean,
                "mean_by_benchmark": mean_by_benchmark,
                "overall_best": best_score,
                "seconds": round(time.time() - generation_started, 3),
            }
            history.append(generation_record)
            print(
                f"GENERATION {generation} COMPLETE | best={generation_best:.4f} | "
                f"mean={generation_mean:.4f} | overall={best_score:.4f} | "
                f"time={generation_record['seconds']:.1f}s",
                flush=True,
            )

            state = {
                "optimizer": optimizer,
                "next_generation": generation + 1,
                "best_score": best_score,
                "best_vector": best_vector,
                "history": history,
                "args": vars(args),
                "agents": [asdict(spec) for spec in DEFAULT_AGENTS],
                "vector_dim": coordinator.vector_dim,
                "run_signature": run_signature,
                "data_manifests": manifests,
            }
            save_pickle_atomic(state, state_path)
            summary_path.write_text(
                json.dumps(
                    {
                        "best_score": best_score,
                        "next_generation": generation + 1,
                        "vector_dim": coordinator.vector_dim,
                        "population_size": optimizer.population_size,
                        "run_signature": run_signature,
                        "data_manifests": manifests,
                        "history": history,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[checkpoint] {state_path}", flush=True)

            if optimizer.should_stop():
                print("[sep-cma] optimizer requested stop", flush=True)
                break

    print("\nTraining finished.")
    print(f"Best batch fitness: {best_score:.4f}")
    print(f"Best coordinator: {best_candidate_path}")
    print(f"Episode log: {log_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. The last completed generation checkpoint is still safe.")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
