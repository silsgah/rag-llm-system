from datetime import datetime

from loguru import logger


class GovernanceMonitor:
    def __init__(self, log_to: str = "logs/governance.log"):
        self.log_file = log_to
        logger.add(self.log_file, rotation="10 MB")

    def check_input(self, text: str) -> bool:
        """
        Apply input compliance checks (forbidden phrases, PII, etc.)
        """
        banned_terms = ["confidential", "password", "classified"]
        if any(term in text.lower() for term in banned_terms):
            logger.warning(f"Governance violation detected in input: {text}")
            return False
        return True

    def check_output(self, text: str) -> str:
        """
        Apply output compliance checks and return status.
        """
        if "confidential" in text.lower():
            logger.warning("Governance violation detected in output.")
            return "non_compliant"
        return "compliant"

    def log_inference(self, prompt: str, output: str, status: str):
        entry = {"timestamp": datetime.utcnow().isoformat(), "input": prompt, "output": output, "status": status}
        logger.info(f"[Governance] {entry}")
