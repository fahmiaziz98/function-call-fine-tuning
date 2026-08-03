# Qwen2.5 Tool-Calling Fine-Tune

A QLoRA fine-tune of `Qwen2.5-1.5B-Instruct` that decides, given a user
request and a set of available tool definitions, whether to call a tool
(and with which arguments) or reply in plain text.

Trained with [Unsloth](https://github.com/unslothai/unsloth) on a single
**NVIDIA A10 GPU**.

### Why fine-tune instead of prompting

Small models often fail at reliable tool selection and argument formatting
under zero-/few-shot prompting alone. Fine-tuning on labeled tool-call
examples turns a small, fast, self-hostable model into a much more
reliable router for this narrow task closing most of the gap to a
larger general-purpose model, but only within tool-calling decisions.

### Scope

- **Single-turn only.** One user request → one decision: call a tool, or
  reply in text. Multi-turn agent planning (consuming a tool's result and
  deciding on a follow-up action) is out of scope for this iteration.
- **Decision + formatting only.** This model decides *whether* and *how*
  to call a tool. It does not execute anything actually running a tool
  and returning its result is the calling application's responsibility.


### Base model & method
| | |
|---|---|
| Base model | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` |
| Method | QLoRA (4-bit base, LoRA adapter) |
| LoRA rank / alpha | 16 / 32 |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj` |
| Trainable params | 4,358,144 / 1,548,072,448 (0.28%) |
| Max sequence length | 1024 |
| Epochs | 3 |
| Effective batch size | 32 (batch 4 × grad accum 8) |
| Precision | bf16 |

### Dataset

Source: **Glaive Function-Calling v2**, parsed into a unified format:

```
system_text: tool schema(s) available for this turn
user_text: the user's request
assistant_text: either a normalized JSON tool call, or a plain-text reply
```

Processing steps:
- **Tool-call detection fix**: Glaive marks tool-call turns with a literal
  `<functioncall>` tag before the JSON payload, not a bare `{` an early
  version of the parser missed this and silently mislabeled every real
  tool-call example as a no-tool example. Fixed by matching on the actual
  tag.
- **Argument normalization**: Glaive's raw `arguments` field is a
  single-quoted JSON *string* nested inside the outer JSON object (not
  valid standard JSON). Re-parsed and re-serialized into a proper nested
  object so the model is trained on clean, directly-parseable output.
- **Near-duplicate removal**: MinHash + LSH deduplication (95% Jaccard
  similarity threshold) over the full example text (system + user +
  assistant combined), since many tool-calling examples share templates
  and differ only in specific argument values.
- **Both response types preserved**: no-tool examples are deliberately
  kept alongside tool-call examples without them, a fine-tuned model
  tends to over-trigger tool calls on every input, regardless of whether
  one is actually needed.

Split 90/5/5 train/val/test, ~37k training examples after deduplication.

### Prompt format

Standard Qwen2.5 ChatML template, tool schemas embedded directly in the
system message:

```
<|im_start|>system
You are a helpful assistant with access to the following functions. Use them if required -
[{"name": "get_current_time", "parameters": {...}}, ...]
<|im_end|>
<|im_start|>user
What time is it in Tokyo right now?
<|im_end|>
<|im_start|>assistant
{"name": "get_current_time", "arguments": {"city": "Tokyo"}}
<|im_end|>
```

For requests that don't need a tool, the assistant turn is a normal text
reply instead of a JSON call.


### Results

**Training**

| Metric | Value |
|---|---|
| Final train loss | 0.372 |
| Final eval loss | 0.384 |
| Final grad_norm | 0.210 |
| Epochs | 3 (3,492 steps) |
| Train runtime | ~108 min on A10 |

Loss dropped quickly in the first ~0.2 epochs (model learning the basic
JSON/no-tool output structure), then converged gradually.
A pattern consistent with a task that has a fairly constrained output format, where
the harder remaining signal is precise argument extraction rather than
format learning.

**Evaluation** (held-out test set)

| Metric | Value |
|---|---|
| Tool-call: valid JSON rate | 0.899 |
| Tool-call: tool name accuracy | 0.899 |
| Tool-call: argument exact match | 0.846 |
| No-tool: abstention accuracy | 0.951 |

Two separate metric categories are tracked because a tool-calling model
can fail in two distinct ways: calling the wrong thing, or calling when it
shouldn't. Abstention accuracy (0.951) shows the model is generally good
at recognizing when *not* to call a tool, which is the harder of the two
failure modes to avoid in practice.

## Quickstart

```python
!git clone https://github.com/fahmiaziz98/function-call-fine-tuning.git
%cd function-call-fine-tuning
!uv pip install --system -q -r requirements.txt

import sys
sys.path.append("./data")
sys.path.append("./src")

import wandb
wandb.login()

from huggingface_hub import login
login()

# 1. Build the dataset
!python data/build_dataset.py --output_dir ./data/processed

# 2. Train
!python src/train.py

# 3. Evaluate
!python src/evaluate.py \
    --checkpoint ./checkpoints/tool-calling \
    --test_file ./data/processed/test.jsonl \
    --run_id <run_id_from_wandb>
```


## Inference

```python
from inference import ToolCallRouter, ToolCall, Reply
from tools import TOOL_SCHEMAS, TOOL_IMPLEMENTATIONS

router = ToolCallRouter("your-username/qwen2.5-1.5b-tool-calling")
result = router.route("What time is it in Tokyo?", TOOL_SCHEMAS)

if isinstance(result, ToolCall):
    output = TOOL_IMPLEMENTATIONS[result.name](**result.arguments)
elif isinstance(result, Reply):
    output = result.text
```

`test_tool_calling.py` runs a small real-world sanity check against 4 mock
tools (simple → complex) plus a couple of no-tool requests useful for
spot-checking behavior beyond aggregate metrics.

---

## Repository structure

```
function-call-fine-tuning/
├── data/
│   ├── build_dataset.py    # parses Glaive, dedupes, logs W&B artifact
│   ├── dedup.py              # MinHash + LSH near-duplicate removal
│   └── schema.py              # ToolCallExample, ResponseType definitions
├── src/
│   ├── config.py                # TrainingConfig
│   ├── train.py                   # Unsloth + SFTTrainer QLoRA loop
│   ├── evaluate.py                 # tool-call + abstention metrics
│   ├── inference.py                 # ToolCallRouter
│   └── tools.py                       # mock tool schemas + implementations
├── test_tool_calling.py                # real-world sanity check script
├── requirements.txt
└── README.md
```

---

## Known limitations

- **Argument exact match (0.846) trails tool name accuracy (0.899).** The
  model more often gets the right tool but slightly wrong or malformed
  arguments than it picks the wrong tool entirely.
- **Occasional malformed JSON on multi-argument calls**, e.g. an observed
  case where the model produced `{"location": "Bandung", , "cuisine":
  "Italian"}` a stray comma making the output invalid JSON. Because
  `ToolCallRouter` only returns a `ToolCall` on successful parsing, this
  case falls back to being treated as a plain-text `Reply` (containing the
  raw malformed JSON as text) rather than crashing but it means the tool
  call is effectively lost rather than executed.
- **A hard case with 5 arguments including a list argument
  (`schedule_meeting`) was not triggered** in one manual test, even though
  all required information was present in the request. The model replied
  conversationally instead of calling the tool — consistent with
  argument-heavy calls being the harder end of this model's ability.
- **Single-turn only.** No evaluation of multi-turn tool use (consuming a
  function's result and deciding on a follow-up call).
