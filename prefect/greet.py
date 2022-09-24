from prefect import flow, task


@task(name="Print Hello")
def print_hello(name: str) -> str:
    msg = f"Hello {name}!"
    print(msg)
    return msg


@flow(name="Subflow")
def my_subflow(msg: str):
    print(f"Subflow says: {msg}")


@flow(name="Greet")
def greet(name: str) -> None:
    message = print_hello(name)
    my_subflow(message)


greet("Hongbo")
