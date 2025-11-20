import os
import subprocess
import sys
from pathlib import Path

import torch  # Needed for detecting GPU count
from huggingface_hub import HfApi
from loguru import logger

from rag_llm_system.settings import settings

finetuning_dir = Path(__file__).resolve().parent
finetuning_requirements_path = finetuning_dir / "requirements.txt"


def run_finetuning_on_sagemaker(
    finetuning_type: str = "sft",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    dataset_huggingface_workspace: str = "gahsilas",
    is_dummy: bool = False,
) -> None:
    """Run finetuning locally but simulate SageMaker environment."""

    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."

    if not finetuning_dir.exists():
        raise FileNotFoundError(f"The directory {finetuning_dir} does not exist.")
    if not finetuning_requirements_path.exists():
        raise FileNotFoundError(f"The file {finetuning_requirements_path} does not exist.")

    # HuggingFace identity check
    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]
    logger.info(f"Current Hugging Face user: {huggingface_user}")

    # Define hyperparameters
    hyperparameters = {
        "finetuning_type": finetuning_type,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "dataset_huggingface_workspace": dataset_huggingface_workspace,
        "model_output_huggingface_workspace": huggingface_user,
    }
    if is_dummy:
        hyperparameters["is_dummy"] = True

    # Prepare local SageMaker-like environment variables
    env = os.environ.copy()
    env.update(
        {
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
            # Fake SageMaker env vars for local runs
            "SM_NUM_GPUS": str(torch.cuda.device_count() if torch.cuda.is_available() else 0),
            "SM_MODEL_DIR": str(finetuning_dir / "outputs"),
            "SM_OUTPUT_DATA_DIR": str(finetuning_dir / "data"),
            "SM_CHANNEL_TRAIN": str(finetuning_dir / "data" / "train"),
        }
    )

    # Ensure output/data dirs exist
    Path(env["SM_MODEL_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["SM_OUTPUT_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["SM_CHANNEL_TRAIN"]).mkdir(parents=True, exist_ok=True)

    # Build finetuning command
    cmd = [sys.executable, "finetune.py"]
    for key, value in hyperparameters.items():
        cmd.extend([f"--{key}", str(value)])

    # Run training (simulated SageMaker)
    subprocess.run(cmd, env=env, cwd=finetuning_dir, check=True)


if __name__ == "__main__":
    run_finetuning_on_sagemaker()
