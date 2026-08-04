import argparse
import os
import sys

from loguru import logger
from unsloth import FastLanguageModel

from config import TrainingConfig

MAX_SEQ_LENGTH = 1024

# Common quantization levels, roughly ordered smallest/lowest-quality to
# largest/highest-quality. q4_k_m is a solid default balance for a 1.5B
# model on modest hardware.
SUPPORTED_QUANTS = ["q4_k_m", "q5_k_m", "q8_0", "f16"]


def _patch_unsloth_gguf_import_bug() -> None:
    """Work around a known Unsloth bug (GitHub issue #5495).

    unsloth_zoo writes llama.cpp's convert_hf_to_gguf.py to a temp file and
    imports it directly, but that script does `from conversion import ...`
    which fails because the temp file's directory isn't on sys.path. This
    patches the loader to add that directory before executing the module.
    """
    import unsloth_zoo.llama_cpp as llama_cpp_module

    original_load = llama_cpp_module._load_module_from_path

    def patched_load(filepath, module_name):
        script_dir = os.path.dirname(filepath)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        return original_load(filepath, module_name)

    llama_cpp_module._load_module_from_path = patched_load


_patch_unsloth_gguf_import_bug()


def export_to_gguf(checkpoint: str, quant: str, config: TrainingConfig) -> None:
    """Load a merged checkpoint and export it to GGUF format.

    Saves locally first (rather than push_to_hub_gguf directly) to avoid a
    known Unsloth bug where the internal temp folder used during
    push_to_hub_gguf sometimes doesn't get config.json written before
    conversion starts.

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
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    local_dir = "./exported/merged_16bit"
    logger.info(f"Saving merged model locally to {local_dir}...")
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)

    # Sanity check before conversion — fail fast with a clear message
    # instead of letting the GGUF converter hit the same missing-file bug.
    config_path = os.path.join(local_dir, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"config.json missing at {local_dir} after save_pretrained — save failed.")

    logger.info(f"Exporting to GGUF (quant={quant})...")
    gguf_dir = "./exported/gguf"
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quant)
    logger.info(f"GGUF export complete. Files written to {gguf_dir}")

    logger.info(f"Pushing GGUF files to {config.hf_repo_gguf}...")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=config.hf_repo_gguf, exist_ok=True)
    api.upload_folder(folder_path=gguf_dir, repo_id=config.hf_repo_gguf)
    logger.info(f"Push complete: https://huggingface.co/{config.hf_repo_gguf}")

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
