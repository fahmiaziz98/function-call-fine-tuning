from unsloth import FastLanguageModel

import argparse
import json
import torch
from collections import defaultdict

import wandb
from loguru import logger

from config import WANDB_PROJECT

MAX_NEW_TOKENS = 256


def load_model(checkpoint: str):
    """Load a fine-tuned model and tokenizer for fast inference.

    Args:
        checkpoint: Local path or HF Hub repo id of the fine-tuned adapter.

    Returns:
        Tuple of (model, tokenizer), with the model set to inference mode.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint, max_seq_length=1024, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response(model, tokenizer, system_text: str, user_text: str) -> str:
    """Generate the model's response for one system+user turn.

    Args:
        model: Fine-tuned model.
        tokenizer: Matching tokenizer.
        system_text: System message (includes tool schema).
        user_text: User request.

    Returns:
        The decoded assistant response text (generated tokens only).
    """
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = output_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def is_tool_call_response(text: str) -> bool:
    """Heuristically check whether a response is a tool call.

    Args:
        text: Generated response text.

    Returns:
        True if the response looks like a JSON tool call.
    """
    return text.strip().startswith("{") and '"name"' in text


def parse_tool_call(text: str) -> dict | None:
    """Parse a tool call response into a dict.

    Args:
        text: Generated response text expected to contain a JSON tool call.

    Returns:
        Parsed dict with "name" and "arguments", or None if parsing fails.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if "name" not in parsed or "arguments" not in parsed:
        return None
    return parsed


def evaluate_tool_call_examples(model, tokenizer, examples: list[dict]) -> dict:
    """Evaluate tool-name accuracy, argument exact-match, and valid-JSON rate.

    Args:
        model: Fine-tuned model.
        tokenizer: Matching tokenizer.
        examples: List of tool_call example dicts.

    Returns:
        Dict with "tool_name_accuracy", "argument_exact_match",
        "valid_json_rate".
    """
    valid_json_count = 0
    correct_name_count = 0
    exact_match_count = 0

    for ex in examples:
        response = generate_response(model, tokenizer, ex["system_text"], ex["user_text"])
        expected = json.loads(ex["assistant_text"])
        predicted = parse_tool_call(response)

        if predicted is None:
            continue

        valid_json_count += 1
        if predicted["name"] == expected["name"]:
            correct_name_count += 1
        if predicted == expected:
            exact_match_count += 1

    total = len(examples)
    return {
        "valid_json_rate": valid_json_count / total if total else 0.0,
        "tool_name_accuracy": correct_name_count / total if total else 0.0,
        "argument_exact_match": exact_match_count / total if total else 0.0,
    }


def evaluate_no_tool_examples(model, tokenizer, examples: list[dict]) -> dict:
    """Evaluate abstention accuracy: did the model correctly avoid a tool call.

    Args:
        model: Fine-tuned model.
        tokenizer: Matching tokenizer.
        examples: List of no_tool example dicts.

    Returns:
        Dict with "abstention_accuracy".
    """
    correct_abstention_count = 0

    for ex in examples:
        response = generate_response(model, tokenizer, ex["system_text"], ex["user_text"])
        if not is_tool_call_response(response):
            correct_abstention_count += 1

    total = len(examples)
    return {"abstention_accuracy": correct_abstention_count / total if total else 0.0}


def load_test_examples(test_file: str) -> dict[str, list[dict]]:
    """Load and group test examples by response_type.

    Args:
        test_file: Path to the test JSONL file.

    Returns:
        Dict mapping "tool_call"/"no_tool" to list of example dicts.
    """
    grouped = defaultdict(list)
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            grouped[row["response_type"]].append(row)
    return grouped


def evaluate_all(checkpoint: str, test_file: str) -> dict:
    """Run full evaluation across both response types.

    Args:
        checkpoint: Path to the fine-tuned model checkpoint.
        test_file: Path to the test JSONL file.

    Returns:
        Dict with "tool_call" and "no_tool" metric sub-dicts.
    """
    model, tokenizer = load_model(checkpoint)
    grouped = load_test_examples(test_file)

    report = {}
    if "tool_call" in grouped:
        logger.info(f"Evaluating {len(grouped['tool_call'])} tool_call examples...")
        report["tool_call"] = evaluate_tool_call_examples(model, tokenizer, grouped["tool_call"])
    if "no_tool" in grouped:
        logger.info(f"Evaluating {len(grouped['no_tool'])} no_tool examples...")
        report["no_tool"] = evaluate_no_tool_examples(model, tokenizer, grouped["no_tool"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned tool-calling model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True, help="W&B run ID (e.g. s2d46sr0)")
    args = parser.parse_args()

    results = evaluate_all(args.checkpoint, args.test_file)
    print(json.dumps(results, indent=2))

    run = wandb.init(project=WANDB_PROJECT, job_type="evaluation", name=f"eval-{args.run_id}")
    for category, metrics in results.items():
        wandb.log({f"{category}/{k}": v for k, v in metrics.items()})
    run.finish()
    logger.info(f"Eval metrics logged as W&B run 'eval-{args.run_id}'")
