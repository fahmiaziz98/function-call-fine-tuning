import argparse

from loguru import logger
from unsloth import FastLanguageModel
from config import TrainingConfig

MAX_SEQ_LENGTH = 1024

# Common quantization levels, roughly ordered smallest/lowest-quality to
# largest/highest-quality. q4_k_m is a solid default balance for a 1.5B
# model on modest hardware.
SUPPORTED_QUANTS = ["q4_k_m", "q5_k_m", "q8_0", "f16"]


def export_to_gguf(checkpoint: str, quant: str, config: TrainingConfig) -> None:
    """Load a merged checkpoint and export it to GGUF format.

    Args:
        checkpoint: HF Hub repo id or local path of the merged 16-bit model.
        quant: Quantization method, one of SUPPORTED_QUANTS.
        config: TrainingConfig object containing export settings.

    Raises:
        ValueError: If `quant` is not a supported quantization method.
    """
    if quant not in SUPPORTED_QUANTS:
        raise ValueError(f"Unsupported quant '{quant}'. Choose from {SUPPORTED_QUANTS}.")

    logger.info(f"Loading checkpoint from {checkpoint}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # already a merged 16-bit checkpoint, not QLoRA
    )
    FastLanguageModel.for_inference(model)  # set to inference mode

    logger.info(f"Exporting to GGUF (quant={quant})...")
    # model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quant)
    model.push_to_hub_gguf(config.hf_repo_gguf, tokenizer, quantization_method=quant)
    logger.info(f"GGUF export complete. Files written to {config.hf_repo_gguf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a merged checkpoint to GGUF.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="HF Hub repo id or local path."
    )
    parser.add_argument(
        "--quant", type=str, default="q4_k_m", choices=SUPPORTED_QUANTS,
        help="Quantization method (default: q4_k_m).",
    )
    args = parser.parse_args()

    export_to_gguf(args.checkpoint, args.quant, TrainingConfig())
