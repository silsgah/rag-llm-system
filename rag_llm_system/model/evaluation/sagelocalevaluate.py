from pathlib import Path
import subprocess
import sys
import os
from huggingface_hub import HfApi
from loguru import logger
from rag_llm_system.settings import settings

evaluation_dir = Path(__file__).resolve().parent
evaluation_requirements_path = evaluation_dir / "requirements.txt"


def run_evaluation_locally(
    is_dummy: bool = False,
    dataset_huggingface_workspace: str = "gahsilas",
) -> None:
    """Run evaluation locally (SageMaker-simulated)."""

    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."

    if not evaluation_dir.exists():
        raise FileNotFoundError(f"The directory {evaluation_dir} does not exist.")
    if not evaluation_requirements_path.exists():
        raise FileNotFoundError(f"The file {evaluation_requirements_path} does not exist.")

    # Identify current Hugging Face user
    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]
    logger.info(f"Current Hugging Face user: {huggingface_user}")

    # Prepare local environment variables
    env = os.environ.copy()
    env.update({
        "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,  # If your eval needs it
        "DATASET_HUGGINGFACE_WORKSPACE": dataset_huggingface_workspace,
        "MODEL_HUGGINGFACE_WORKSPACE": huggingface_user,
        # You can add more if evaluation.py expects them
    })
    if is_dummy:
        env["IS_DUMMY"] = "True"

    # Ensure outputs folder exists
    Path(evaluation_dir / "outputs").mkdir(parents=True, exist_ok=True)

    # Build evaluation command (like fine-tuning)
    cmd = [sys.executable, "evaluate.py"]

    # Run evaluation script locally
    subprocess.run(cmd, env=env, cwd=evaluation_dir, check=True)


if __name__ == "__main__":
    run_evaluation_locally()
