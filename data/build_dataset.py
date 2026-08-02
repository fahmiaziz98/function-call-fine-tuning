import argparse
import json
import random
import re
import sys
from pathlib import Path

import wandb
from datasets import load_dataset
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import WANDB_PROJECT  # noqa: E402

from dedup import deduplicate_indices
from schema import ResponseType, ToolCallExample

RANDOM_SEED = 42
VAL_RATIO = 0.05
TEST_RATIO = 0.05
DEDUP_THRESHOLD = 0.95
DATASET_ARTIFACT_NAME = "tool-calling-dataset"

# Glaive marks no-tool-call turns with this literal string in its
# "chat" field structure; adjust if using a different source dataset.
NO_FUNCTION_CALL_MARKER = "none"
TOOL_CALL_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*"arguments"\s*:\s*\'(?P<arguments>.*)\'\s*\}',
    re.DOTALL,
)


def normalize_tool_call(raw_call: str) -> str | None:
    """Normalize a Glaive-style tool call into clean, consistent JSON.

    Glaive wraps the "arguments" value as a single-quoted JSON string
    inside the outer JSON object, which is not valid standard JSON. This
    parses that structure and re-serializes it as a proper nested object,
    so the model is trained on consistent, directly-parseable output.

    Args:
        raw_call: Raw tool call text, e.g.
            '{"name": "x", "arguments": \'{"a": 1}\'}'.

    Returns:
        A clean JSON string, or None if the pattern doesn't match (row
        should be skipped as malformed).
    """
    match = TOOL_CALL_PATTERN.search(raw_call)
    if not match:
        return None

    try:
        arguments = json.loads(match.group("arguments"))
    except json.JSONDecodeError:
        return None

    return json.dumps({"name": match.group("name"), "arguments": arguments}, ensure_ascii=False)


def parse_glaive_row(row: dict) -> ToolCallExample | None:
    """Parse a single Glaive Function-Calling v2 row into a ToolCallExample.

    Only the first user/assistant turn is extracted, even if the raw
    conversation has multiple turns — this matches the project's scope of
    single-turn tool-call decisions (see SYSTEM_DESIGN.md Section 2:
    "Non-goals"). Later turns (e.g. a follow-up FUNCTION RESPONSE and the
    natural-language summary after it) are intentionally dropped.

    Args:
        row: A raw row from the Glaive dataset, expected to have "system",
            "chat" fields following Glaive's function-calling format.

    Returns:
        A ToolCallExample, or None if the row is malformed and should be
        skipped.
    """
    system_text = row.get("system", "").strip()
    system_text = re.sub(r"^SYSTEM:\s*", "", system_text)
    chat_text = row.get("chat", "").strip()
    if not system_text or not chat_text:
        return None

    user_match = re.search(r"USER:\s*(.*?)\s*(?:ASSISTANT:|$)", chat_text, re.DOTALL)
    assistant_match = re.search(r"ASSISTANT:\s*(.*?)(?:<\|endoftext\|>|$)", chat_text, re.DOTALL)
    if not user_match or not assistant_match:
        return None

    user_text = user_match.group(1).strip()
    assistant_text = assistant_match.group(1).strip()
    if not user_text or not assistant_text:
        return None

    # Glaive marks tool-call responses with a literal "<functioncall>" tag
    # before the JSON payload — NOT a bare "{" as previously assumed. That
    # earlier check silently misclassified every real tool-call example as
    # NO_TOOL, since none of them start with "{" directly.
    is_tool_call = assistant_text.startswith("<functioncall>")

    if is_tool_call:
        # Strip the tag, keep just the JSON call payload as the target.
        assistant_text = assistant_text.removeprefix("<functioncall>").strip()
        normalized = normalize_tool_call(assistant_text)
        if not normalized:
            return None
        assistant_text = normalized

    response_type = ResponseType.TOOL_CALL if is_tool_call else ResponseType.NO_TOOL

    return ToolCallExample(
        response_type=response_type,
        system_text=system_text,
        user_text=user_text,
        assistant_text=assistant_text,
    )


def build_examples(raw_dataset) -> list[ToolCallExample]:
    """Parse all rows into ToolCallExample objects, skipping malformed ones.

    Args:
        raw_dataset: The raw HuggingFace dataset split.

    Returns:
        List of successfully parsed ToolCallExample objects.
    """
    examples = []
    skipped = 0
    for row in raw_dataset:
        example = parse_glaive_row(row)
        if example is None:
            skipped += 1
            continue
        examples.append(example)

    logger.info(f"Parsed {len(examples)} examples ({skipped} skipped as malformed)")
    return examples


def report_response_type_balance(examples: list[ToolCallExample]) -> None:
    """Log the ratio of tool-call vs no-tool examples.

    Args:
        examples: Examples to report on.
    """
    tool_call_count = sum(1 for e in examples if e.response_type == ResponseType.TOOL_CALL)
    no_tool_count = len(examples) - tool_call_count
    logger.info(
        f"Response type balance -> tool_call: {tool_call_count}, no_tool: {no_tool_count} "
        f"({no_tool_count / len(examples):.1%} no-tool)"
    )


def deduplicate_examples(
    examples: list[ToolCallExample], threshold: float
) -> list[ToolCallExample]:
    """Remove near-duplicate examples based on full example text.

    Args:
        examples: Examples to deduplicate.
        threshold: Jaccard similarity threshold for near-duplicate removal.

    Returns:
        Deduplicated list of examples, preserving original order.
    """
    full_texts = [e.full_text() for e in examples]
    keep_indices = deduplicate_indices(full_texts, threshold=threshold)
    return [examples[i] for i in keep_indices]


def split_examples(
    examples: list[ToolCallExample], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[ToolCallExample], list[ToolCallExample], list[ToolCallExample]]:
    """Shuffle and split examples into train/val/test subsets.

    Args:
        examples: Examples to split.
        val_ratio: Fraction reserved for validation.
        test_ratio: Fraction reserved for test.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) example lists.
    """
    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    val = shuffled[:n_val]
    test = shuffled[n_val : n_val + n_test]
    train = shuffled[n_val + n_test :]
    return train, val, test


def write_jsonl(examples: list[ToolCallExample], path: Path) -> None:
    """Write a list of examples to a JSONL file.

    Args:
        examples: Examples to write.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def log_dataset_artifact(output_path: Path, counts: dict) -> None:
    """Log the processed dataset directory as a versioned W&B Artifact.

    Args:
        output_path: Directory containing train/val/test JSONL files.
        counts: Dict with per-split example counts, stored as metadata.
    """
    run = wandb.init(project=WANDB_PROJECT, job_type="build-dataset")
    artifact = wandb.Artifact(
        name=DATASET_ARTIFACT_NAME,
        type="dataset",
        metadata={
            "source": "glaive-function-calling-v2",
            "dedup_threshold": DEDUP_THRESHOLD,
            **counts,
        },
    )
    artifact.add_dir(str(output_path))
    run.log_artifact(artifact)
    run.finish()
    logger.info(f"Dataset logged as W&B Artifact '{DATASET_ARTIFACT_NAME}'")


def main(output_dir: str, max_samples: int | None = None) -> None:
    """Build the tool-calling dataset end-to-end and log it to W&B.

    Args:
        output_dir: Directory where the resulting JSONL files are written.
        max_samples: If set, only the first `max_samples` raw rows are
            loaded and processed. Useful for quickly testing the pipeline
            (parsing, dedup, splitting) without waiting on the full
            dataset. Leave unset for a real training run.
    """
    output_path = Path(output_dir)

    logger.info("Loading Glaive Function-Calling v2...")
    raw_dataset = load_dataset("glaiveai/glaive-function-calling-v2")["train"]

    if max_samples is not None:
        raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))
        logger.info(f"max_samples set -> using {len(raw_dataset)} raw rows")

    logger.info("Parsing examples...")
    examples = build_examples(raw_dataset)
    report_response_type_balance(examples)

    logger.info(f"Deduplicating at threshold={DEDUP_THRESHOLD}...")
    examples = deduplicate_examples(examples, DEDUP_THRESHOLD)
    report_response_type_balance(examples)

    logger.info("Splitting into train/val/test...")
    train, val, test = split_examples(examples, VAL_RATIO, TEST_RATIO, RANDOM_SEED)

    write_jsonl(train, output_path / "train.jsonl")
    write_jsonl(val, output_path / "val.jsonl")
    write_jsonl(test, output_path / "test.jsonl")

    counts = {"train_count": len(train), "val_count": len(val), "test_count": len(test)}
    logger.info(f"Done. {counts}")

    # log_dataset_artifact(output_path, counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the tool-calling dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Directory to write train/val/test JSONL files.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, only process the first N raw rows (for quick local testing).",
    )
    args = parser.parse_args()
    main(args.output_dir, args.max_samples)
