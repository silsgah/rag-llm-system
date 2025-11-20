from zenml import step

from rag_llm_system.model.evaluation.sagelocalevaluate import run_evaluation_locally


@step
def evaluate(
    is_dummy: bool = False,
) -> None:
    run_evaluation_locally(
        is_dummy=is_dummy,
    )
