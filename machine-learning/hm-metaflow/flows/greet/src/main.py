# ruff: noqa: T201
from typing import Protocol

from metaflow import FlowSpec, Parameter, step


class JoinInputs(Protocol):
    process_data: "GreetFlow"
    calculate: "GreetFlow"


class GreetFlow(FlowSpec):
    message: str
    numbers: list[int]
    processed_numbers: list[int]
    sum_value: int
    avg_value: float

    greeting: str = Parameter(
        "name",
        default="World",
    )

    multiplier: int = Parameter(
        "multiplier",
        default=2,
    )

    @step
    def start(self) -> None:
        print(f"Greeting: {self.greeting}")
        self.message: str = f"Hello, {self.greeting}!"
        self.numbers: list[int] = [1, 2, 3, 4, 5]
        self.next(self.process_data, self.calculate)

    @step
    def process_data(self) -> None:
        print("Processing data...")
        self.processed_numbers: list[int] = [n * self.multiplier for n in self.numbers]
        print(f"Processed numbers: {self.processed_numbers}")
        self.next(self.join)

    @step
    def calculate(self) -> None:
        print("Calculating statistics...")
        self.sum_value: int = sum(self.numbers)
        self.avg_value: float = self.sum_value / len(self.numbers)
        print(f"Sum: {self.sum_value}, Average: {self.avg_value}")
        self.next(self.join)

    @step
    def join(self, inputs: JoinInputs) -> None:
        print("Joining branches...")
        self.message: str = inputs.process_data.message
        self.numbers: list[int] = inputs.process_data.numbers
        self.processed_numbers: list[int] = inputs.process_data.processed_numbers
        self.sum_value: int = inputs.calculate.sum_value
        self.avg_value: float = inputs.calculate.avg_value
        self.next(self.end)

    @step
    def end(self) -> None:
        print(self.message)
        print(f"Original numbers: {self.numbers}")
        print(f"Processed numbers: {self.processed_numbers}")
        print(f"Sum: {self.sum_value}, Average: {self.avg_value}")
        print("Flow completed successfully!")


if __name__ == "__main__":
    GreetFlow()
