import wandb
from datasets import load_dataset
from loguru import logger
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from config import WANDB_PROJECT, TrainingConfig


def load_model_and_tokenizer(config: TrainingConfig):
    """Load the base model in 4-bit and attach a LoRA adapter.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (model, tokenizer) ready for training.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    return model, tokenizer


def format_example(row: dict, tokenizer) -> dict:
    """Format one raw example into a chat-templated training text.

    Args:
        row: Dict with "system_text", "user_text", "assistant_text".
        tokenizer: Tokenizer providing the chat template.

    Returns:
        Dict with a single "text" field, ready for SFTTrainer.
    """
    messages = [
        {"role": "system", "content": row["system_text"]},
        {"role": "user", "content": row["user_text"]},
        {"role": "assistant", "content": row["assistant_text"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def build_dataset(data_dir: str, tokenizer):
    """Load JSONL splits and apply chat-template formatting.

    Args:
        data_dir: Directory containing train.jsonl and val.jsonl.
        tokenizer: Tokenizer used for chat-template formatting.

    Returns:
        A HuggingFace DatasetDict with "train" and "validation" splits.
    """
    data_files = {"train": f"{data_dir}/train.jsonl", "validation": f"{data_dir}/val.jsonl"}
    raw_dataset = load_dataset("json", data_files=data_files)
    return raw_dataset.map(lambda row: format_example(row, tokenizer))


def train(config: TrainingConfig) -> None:
    """Run the full QLoRA fine-tuning loop with W&B tracking and lineage.

    Args:
        config: Training configuration.
    """
    run = wandb.init(
        project=WANDB_PROJECT, name=config.run_name, config=config.__dict__, job_type="train"
    )

    dataset_artifact = run.use_artifact(config.dataset_artifact)
    dataset_dir = dataset_artifact.download()

    model, tokenizer = load_model_and_tokenizer(config)
    dataset = build_dataset(dataset_dir, tokenizer)

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        seed=config.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=20,
        report_to=["wandb"],
        run_name=config.run_name,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    adapter_artifact = wandb.Artifact(
        name="tool-calling-adapter",
        type="model",
        metadata={"base_model": config.model_name, "run_name": config.run_name},
    )
    # adapter_artifact.add_dir(config.output_dir)
    run.log_artifact(adapter_artifact)

    if config.push_to_hub:
        model.push_to_hub(config.hf_repo_id)
        tokenizer.push_to_hub(config.hf_repo_id)
        logger.info(f"Adapter pushed to https://huggingface.co/{config.hf_repo_id}")

    run.finish()
    logger.info(f"Training complete. Adapter saved to {config.output_dir}")


if __name__ == "__main__":
    train(TrainingConfig())
