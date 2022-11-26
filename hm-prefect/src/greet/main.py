from prefect import flow, task
from pydantic import BaseModel


class User(BaseModel):
    first_name: str
    last_name: str


@task
def print_hello(name: str) -> str:
    msg = f"Hello {name}!"
    print(msg)
    return msg


@flow
def my_subflow(msg: str):
    print(f"Subflow says: {msg}")


@flow
def greet(user: User) -> None:
    message = print_hello(f"{user.first_name} {user.last_name}")
    my_subflow(message)


if __name__ == "__main__":
    external_user = User(first_name="Hongbo", last_name="Miao")
    greet(external_user)
