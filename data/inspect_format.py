import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"


def load_samples(data_dir: str, num_samples: int) -> list[dict]:
    """Load the first N examples from train.jsonl.

    Args:
        data_dir: Directory containing train.jsonl.
        num_samples: Number of examples to load.

    Returns:
        List of raw example dicts.
    """
    path = Path(data_dir) / "train.jsonl"
    samples = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))
    return samples


def inspect(data_dir: str, num_samples: int) -> None:
    """Print raw fields and formatted chat-template text for each sample.

    Args:
        data_dir: Directory containing train.jsonl.
        num_samples: Number of examples to inspect.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    samples = load_samples(data_dir, num_samples)

    for i, row in enumerate(samples, start=1):
        messages = [
            {"role": "system", "content": row["system_text"]},
            {"role": "user", "content": row["user_text"]},
            {"role": "assistant", "content": row["assistant_text"]},
        ]
        formatted_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        token_ids = tokenizer(formatted_text)["input_ids"]
        decoded_back = tokenizer.decode(token_ids)

        print(f"\n{'=' * 70}")
        print(f"Sample {i} | response_type: {row['response_type']} | token count: {len(token_ids)}")
        print(f"{'=' * 70}")
        print(f"--- assistant_text (raw target) ---\n{row['assistant_text']}")
        print(f"\n--- formatted_text (fed to SFTTrainer) ---\n{formatted_text}")
        print(f"\n--- decoded_back (round-trip check) ---\n{decoded_back}")
        print(f"\n--- round-trip matches formatted_text: {decoded_back.strip() == formatted_text.strip()} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect chat-template formatting before training.")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    inspect(args.data_dir, args.num_samples)
