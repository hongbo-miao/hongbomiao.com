from pydantic import BaseModel

from prefect import flow, task


class User(BaseModel):
    first_name: str
    last_name: str


@task(name="Print Hello")
def print_hello(name: str) -> str:
    msg = f"Hello {name}!"
    print(msg)
    return msg


@flow(name="Subflow")
def my_subflow(msg: str):
    print(f"Subflow says: {msg}")


@flow(name="Greet")
def greet(user: User) -> None:
    message = print_hello(f"{user.first_name} {user.last_name}")
    my_subflow(message)


if __name__ == "__main__":
    external_user = User(first_name="Hongbo", last_name="Miao")
    greet(external_user)
