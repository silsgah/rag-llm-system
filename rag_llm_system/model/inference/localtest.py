# from loguru import logger
# from rag_llm_system.model.inference.run import InferenceExecutor
# from rag_llm_system.model.inference.local import LLMInferenceLocalEndpoint

# if __name__ == "__main__":
#     text = "Write me a LinkedIn post about running local LLM inference with GPU."
#     logger.info(f"Running inference for text: '{text}'")

#     llm = LLMInferenceLocalEndpoint(endpoint_url="http://127.0.0.1:8000/infer")
#     answer = InferenceExecutor(llm, text).execute()

#     logger.info(f"Answer: '{answer}'")


import opik
from loguru import logger
from opik import opik_context

from rag_llm_system import settings
from rag_llm_system.application.utils import misc
from rag_llm_system.infrastructure.opik_utils import configure_opik
from rag_llm_system.model.inference.local import LLMInferenceLocalEndpoint
from rag_llm_system.model.inference.run import InferenceExecutor

# 🔧 Configure Opik (links to your Comet workspace)
configure_opik()


@opik.track
def local_inference_test(prompt: str):
    logger.info(f"Running inference for text: '{prompt}'")

    llm = LLMInferenceLocalEndpoint(endpoint_url="http://127.0.0.1:8000/infer")
    answer = InferenceExecutor(llm, prompt).execute()

    # 📊 Log metadata to Opik
    opik_context.update_current_trace(
        tags=["local_inference", "gpu", "rag-llm-system"],
        metadata={
            "model_id": settings.HF_MODEL_ID,
            "device": "cuda",
            "temperature": settings.TEMPERATURE_INFERENCE,
            "prompt_tokens": misc.compute_num_tokens(prompt),
            "answer_tokens": misc.compute_num_tokens(answer),
        },
    )

    logger.info(f"Answer: '{answer}'")
    return answer


if __name__ == "__main__":
    text = "The recent amendment of the agricultural agreement between Morocco and the European Union signifies a noteworthy development in international trade relations. This agreement confirms the applicability of preferential tariffs to Southern Provinces,"
    local_inference_test(text)
