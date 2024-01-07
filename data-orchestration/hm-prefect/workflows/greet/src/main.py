from prefect import flow, get_run_logger, task
from pydantic import BaseModel


class User(BaseModel):
    first_name: str
    last_name: str


@task
def get_hello(name: str) -> str:
    return f"Hello {name}!"


@flow
def greet_subflow(msg: str):
    logger = get_run_logger()
    logger.info(f"Subflow says: {msg}")


@flow
def greet(user: User) -> None:
    logger = get_run_logger()
    message = get_hello(f"{user.first_name} {user.last_name}")
    logger.info(message)
    greet_subflow(message)


if __name__ == "__main__":
    external_user = User(first_name="Hongbo", last_name="Miao")
    greet(external_user)
