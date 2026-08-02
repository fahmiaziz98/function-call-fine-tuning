import uuid
from dataclasses import dataclass, field

WANDB_PROJECT = "qwen-tool-calling"


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for QLoRA fine-tuning.

    Attributes:
        model_name: Base model to fine-tune.
        max_seq_length: Max sequence length (tool schemas can be long).
        load_in_4bit: Whether to load the base model in 4-bit (QLoRA).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_target_modules: Which attention projections to apply LoRA to.
        output_dir: Local directory where the adapter is saved.
        per_device_train_batch_size: Batch size per device during training.
        gradient_accumulation_steps: Steps to accumulate before an optimizer step.
        learning_rate: Optimizer learning rate.
        num_train_epochs: Number of full passes over the training data.
        warmup_steps: Linear warmup steps for the LR scheduler.
        seed: Random seed for reproducibility.
        run_name: Unique identifier for this run, used as the W&B run id.
        dataset_artifact: W&B artifact reference for the training dataset.
        push_to_hub: Whether to push the final adapter to the HF Hub.
        hf_repo_id: Target HF Hub repo id, used only if push_to_hub is True.
    """

    model_name: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    max_seq_length: int = 1024
    load_in_4bit: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    output_dir: str = "./checkpoints/tool-calling"

    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_steps: int = 50
    seed: int = 42

    run_name: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    dataset_artifact: str = "tool-calling-dataset:latest"

    push_to_hub: bool = False
    hf_repo_id: str = "fahmiaziz/qwen2.5-1.5b-tool-calling"
