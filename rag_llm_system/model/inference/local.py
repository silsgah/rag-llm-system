import requests
from loguru import logger


class LLMInferenceLocalEndpoint:
    """
    Mimics the SageMaker LLM inference endpoint but calls the local FastAPI server.
    """

    def __init__(self, endpoint_url: str = "http://127.0.0.1:8000/infer"):
        self.endpoint_url = endpoint_url

    def set_payload(self, inputs: str, parameters: dict | None = None):
        """
        For compatibility with InferenceExecutor. Parameters are optional.
        """
        self.payload = {"prompt": inputs}

    def inference(self):
        """
        Sends the prompt to the local API and returns a list with a dict containing 'generated_text'.
        """
        logger.info(f"Sending prompt to local inference API: {self.payload['prompt'][:80]}...")
        response = requests.post(self.endpoint_url, json=self.payload)
        response.raise_for_status()
        data = response.json()
        return [{"generated_text": data["generated_text"]}]
