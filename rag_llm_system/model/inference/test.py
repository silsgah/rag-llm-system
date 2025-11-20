from loguru import logger

from rag_llm_system.model.inference.inference import LLMInferenceSagemakerEndpoint
from rag_llm_system.model.inference.run import InferenceExecutor
from rag_llm_system.settings import settings

if __name__ == "__main__":
    text = "Write me a post about AWS SageMaker inference endpoints."
    logger.info(f"Running inference for text: '{text}'")
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, text).execute()

    logger.info(f"Answer: '{answer}'")
