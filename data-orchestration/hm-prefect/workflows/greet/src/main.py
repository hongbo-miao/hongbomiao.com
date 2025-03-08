from prefect import flow, get_run_logger
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


@flow
def create_greeting_subflow(name: str) -> str:
    logger = get_run_logger()
    greeting = f"Hello {name}!"
    logger.info(greeting)
    return greeting


@flow
def create_farewell_subflow(age: int) -> str:
    logger = get_run_logger()
    message = "Good night!" if age > 50 else "Goodbye!"
    logger.info(message)
    return message


@flow
def hm_greet(user: User) -> None:
    greeting = create_greeting_subflow(user.name)
    farewell = create_farewell_subflow(user.age)
    logger = get_run_logger()
    logger.info(f"Final messages: {greeting} {farewell}")


if __name__ == "__main__":
    external_user = User(name="Rose", age=20)
    hm_greet(external_user)
