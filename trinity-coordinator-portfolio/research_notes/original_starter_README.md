# TRINITY Reproduction Starter

This is a practical starter implementation of a TRINITY-style LLM coordinator.

It implements:

- a small coordinator backbone interface
- a lightweight linear routing head
- model selection logits + role selection logits
- three roles: Thinker, Worker, Verifier
- a multi-turn runtime loop
- binary reward checking for math-style tasks
- a simple diagonal evolution strategy trainer
- optional hooks for Hugging Face local models

The goal is to reproduce the **process/system**, not the exact paper numbers.
Exact paper-level reproduction needs their supplementary source code, trained weights, APIs, benchmark splits, and compute budget.

## Architecture

For each question:

```text
question + previous transcript
    -> coordinator SLM hidden state
    -> linear head
    -> choose agent + role
    -> role prompt
    -> selected LLM/agent response
    -> append to transcript
    -> repeat until verifier accepts or max turns reached
```

## Folder structure

```text
trinity_repro_starter/
  coordinator/
    backbone.py       # DummyBackbone and QwenBackbone
    head.py           # LinearCoordinationHead
    router.py         # argmax/sample routing
  agents/
    base.py           # common Agent interface
    toy_agents.py     # cheap local toy agents for debugging
    hf_agent.py       # optional Hugging Face chat model agent
  prompts/
    roles.py          # Thinker / Worker / Verifier prompt templates
  runtime/
    trinity_loop.py   # multi-turn coordination loop
  train/
    reward.py         # math reward checker
    diag_es.py        # simple diagonal ES optimizer
    cma_train.py      # head training loop
  examples/
    tiny_math.jsonl   # small toy dataset
  run_demo.py         # runs one question with toy agents
  train_toy.py        # trains the toy version cheaply
```

## Quick start

From inside this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
python train_toy.py
```

The demo uses `DummyBackbone`, `ScriptedHead`, and `ToyMathAgent`, so it does not need GPUs or API keys. `ScriptedHead` is only for checking the loop; training and real experiments use `LinearCoordinationHead`.

## Using a real Qwen coordinator

Install the heavier dependencies:

```bash
pip install torch transformers accelerate sentencepiece
```

Then replace `DummyBackbone` with:

```python
from coordinator.backbone import QwenBackbone
backbone = QwenBackbone(model_name="Qwen/Qwen3-0.6B", device="cuda")
```

## Using real LLM agents

Use `HFChatAgent` for local Hugging Face models:

```python
from agents.hf_agent import HFChatAgent
agents = [
    HFChatAgent("Qwen/Qwen3-32B", device="cuda"),
]
```

For GPT/Gemini/Claude, add your own API wrapper by subclassing `Agent` in `agents/base.py`.

## Recommended reproduction order

1. Run `run_demo.py`.
2. Confirm hidden vector -> head logits -> role/model choice works.
3. Train the toy version with `train_toy.py`.
4. Swap in `QwenBackbone`.
5. Add one real local LLM agent.
6. Add more agents.
7. Train on Math500 first.
8. Add MMLU.
9. Add LiveCodeBench/code execution last.
10. Add singular value fine-tuning only after the routing loop works.

## Important caveats

- `diag_es.py` is a simple diagonal evolution strategy. It is not a full implementation of all sep-CMA-ES details.
- For closer paper reproduction, install the `cma` Python package and replace the optimizer with `cma.CMAEvolutionStrategy(..., {"CMA_diagonal": True})`.
- Paper-level training can be very expensive: many candidate heads × many evaluation tasks × up to several LLM calls per task.
