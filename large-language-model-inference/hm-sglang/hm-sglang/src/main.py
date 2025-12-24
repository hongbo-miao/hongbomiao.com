import logging

import sglang as sgl
from sglang.lang.ir import SglExpr

logger = logging.getLogger(__name__)


@sgl.function
def ask_question(s: SglExpr, question: str) -> None:
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=1024))


def get_answer(question: str) -> str:
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:8000"))
    state = ask_question.run(question=question)
    return state["answer"]


def main() -> None:
    question = "What is gravity?"
    answer = get_answer(question)
    logger.info(f"Q: {question}")
    logger.info(f"A: {answer}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
