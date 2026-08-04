"""Export a merged 16-bit checkpoint to GGUF manually via llama.cpp,
bypassing Unsloth's built-in GGUF export (which has multiple internal
bugs in this version — see conversation history / GitHub issue #5495).
"""

import os
import subprocess

from huggingface_hub import HfApi, snapshot_download
from loguru import logger

MERGED_LOCAL_DIR = "./exported/merged_16bit"
GGUF_OUTPUT_PATH = "./exported/gguf/model.gguf"
LLAMA_CPP_DIR = "./llama.cpp"


def download_merged_model(repo_id: str) -> str:
    """Download the merged 16-bit checkpoint from the HF Hub.

    Args:
        repo_id: HF Hub repo id of the merged model.

    Returns:
        Local path where the model was downloaded.
    """
    logger.info(f"Downloading {repo_id}...")
    local_path = snapshot_download(repo_id=repo_id, local_dir=MERGED_LOCAL_DIR)
    logger.info(f"Downloaded to {local_path}")
    return local_path


def setup_llama_cpp() -> None:
    """Clone llama.cpp and install required dependencies."""

    if not os.path.exists(LLAMA_CPP_DIR):
        logger.info("Cloning llama.cpp...")

        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/ggml-org/llama.cpp",
                LLAMA_CPP_DIR,
            ],
            check=True,
        )

    logger.info("Installing build dependencies...")

    subprocess.run(
        [
            "apt-get",
            "update",
        ],
        check=True,
    )

    subprocess.run(
        [
            "apt-get",
            "install",
            "-y",
            "cmake",
            "build-essential",
        ],
        check=True,
    )

    logger.info("Installing Python requirements...")

    subprocess.run(
        [
            "pip",
            "install",
            "-q",
            "-r",
            f"{LLAMA_CPP_DIR}/requirements.txt",
        ],
        check=True,
    )


def convert_to_gguf_f16(model_dir: str, output_path: str) -> None:
    """Convert HF model to base GGUF format (f16), the required first step
    before K-quant quantization.

    Args:
        model_dir: Local directory containing the merged 16-bit model.
        output_path: Destination .gguf file path (f16, unquantized).

    Raises:
        subprocess.CalledProcessError: If the conversion script fails.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Converting HF model to base GGUF (f16)...")
    subprocess.run(
        [
            "python", f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py",
            model_dir,
            "--outfile", output_path,
            "--outtype", "f16",
        ],
        check=True,
    )
    logger.info(f"Base GGUF (f16) written to {output_path}")


def build_llama_quantize_binary() -> str:
    """Build llama-quantize from source if it does not already exist.

    Returns:
        Path to the llama-quantize executable.

    Raises:
        RuntimeError: If the binary cannot be found after compilation.
        subprocess.CalledProcessError: If CMake configuration/build fails.
    """
    import shutil

    build_dir = os.path.join(LLAMA_CPP_DIR, "build")

    possible_paths = [
        os.path.join(build_dir, "bin", "llama-quantize"),
        os.path.join(build_dir, "bin", "Release", "llama-quantize"),
        os.path.join(build_dir, "llama-quantize"),
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            logger.info(f"Using existing llama-quantize: {path}")
            return path

    logger.info("Building llama-quantize...")

    if shutil.which("cmake") is None:
        raise RuntimeError(
            "cmake is not installed.\n"
            "Install it first:\n"
            "apt-get update && apt-get install -y cmake build-essential"
        )

    subprocess.run(
        [
            "cmake",
            "-S",
            LLAMA_CPP_DIR,
            "-B",
            build_dir,
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
        ],
        check=True,
    )

    subprocess.run(
        [
            "cmake",
            "--build",
            build_dir,
            "--target",
            "llama-quantize",
            "-j2",
        ],
        check=True,
    )

    for path in possible_paths:
        if os.path.isfile(path):
            logger.info(f"Built llama-quantize: {path}")
            return path

    raise RuntimeError(
        "llama-quantize was successfully built but its location "
        "could not be determined."
    )


def quantize_gguf(f16_path: str, quantized_path: str, quant: str) -> None:
    """Quantize a base f16 GGUF file to a K-quant format (e.g. q4_k_m).

    Args:
        f16_path: Path to the base f16 .gguf file.
        quantized_path: Destination path for the quantized .gguf file.
        quant: Target quantization type, e.g. "Q4_K_M" (llama-quantize
            expects uppercase).

    Raises:
        subprocess.CalledProcessError: If quantization fails.
    """
    binary_path = build_llama_quantize_binary()
    logger.info(f"Quantizing to {quant}...")
    subprocess.run([binary_path, f16_path, quantized_path, quant.upper()], check=True)
    logger.info(f"Quantized GGUF written to {quantized_path}")


def push_gguf_to_hub(gguf_path: str, repo_id: str) -> None:
    """Upload the GGUF file to a HF Hub repo.

    Args:
        gguf_path: Local path to the .gguf file.
        repo_id: Target HF Hub repo id.
    """
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_file(
        path_or_fileobj=gguf_path,
        path_in_repo=os.path.basename(gguf_path),
        repo_id=repo_id,
    )
    logger.info(f"Pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manually export a checkpoint to GGUF.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hf_repo_gguf", type=str, required=True)
    parser.add_argument("--quant", type=str, default="q4_k_m")
    args = parser.parse_args()

    model_dir = download_merged_model(args.checkpoint)
    setup_llama_cpp()

    f16_path = "./exported/gguf/model-f16.gguf"
    quantized_path = f"./exported/gguf/model-{args.quant}.gguf"

    convert_to_gguf_f16(model_dir, f16_path)
    quantize_gguf(f16_path, quantized_path, args.quant)
    push_gguf_to_hub(quantized_path, args.hf_repo_gguf)
