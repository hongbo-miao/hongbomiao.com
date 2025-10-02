from datetime import UTC, datetime

from airflow.models.taskinstance import TaskInstance
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, task, task_group

with DAG(
    "greet",
    start_date=datetime(2022, 1, 1, tzinfo=UTC),
    schedule="@once",
    catchup=False,
    params={
        "first_name": "Hongbo",
        "last_name": "Miao",
    },
) as dag:

    @task(task_id="get_name")
    def get_name(params: dict[str, str] | None = None) -> dict[str, str]:
        if params is None:
            message = "params cannot be None"
            raise ValueError(message)
        return {
            "first_name": params["first_name"],
            "last_name": params["last_name"],
        }

    @task(task_id="transform_first_name")
    def transform_first_name(name: dict[str, str]) -> str:
        return name["first_name"][0]

    @task
    def transform_last_name(name: dict[str, str]) -> str:
        return name["last_name"][0]

    @task
    def get_initials(first_name_initial: str, last_name_initial: str) -> str:
        return f"{first_name_initial}.{last_name_initial}."

    get_time = BashOperator(
        task_id="get_time",
        bash_command="date '+%Y-%m-%d %H:%M:%S'",
    )

    @task
    def greet(initials: str, ti: TaskInstance | None = None) -> str:
        if ti is None:
            message = "TaskInstance cannot be None"
            raise ValueError(message)
        time = ti.xcom_pull(task_ids="get_time")
        if time is None:
            message = "Failed to get time from xcom"
            raise ValueError(message)
        dt = datetime.strptime(time, "%Y-%m-%d %H:%M:%S").astimezone(UTC)
        return f"Hello {initials}, {dt.strftime('at %H:%M on %B %d, %Y')}!"

    @task_group
    def initials_group_function(name: dict[str, str]) -> str:
        return get_initials(transform_first_name(name), transform_last_name(name))

    my_name = get_name()
    my_initials = initials_group_function(my_name)
    run_greet = greet(my_initials)

    [my_initials, get_time] >> run_greet
