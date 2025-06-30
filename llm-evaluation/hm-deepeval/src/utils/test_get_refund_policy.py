from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from utils.get_refund_policy import get_refund_policy


def test_get_refund_policy() -> None:
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
    )
    question = "What if these shoes don't fit?"
    test_case = LLMTestCase(
        retrieval_context=[
            "All customers are eligible for a 30 day full refund at no extra costs.",
        ],
        input=question,
        actual_output=get_refund_policy(question, refund_day_number=30),
        expected_output="We offer a 30-day full refund at no extra costs.",
    )
    assert_test(test_case, [correctness_metric])
