from datetime import datetime
from typing import Dict

from airflow import DAG
from airflow.decorators import task, task_group
from airflow.operators.bash import BashOperator

with DAG(
    "greet",
    start_date=datetime(2022, 1, 1),
    schedule_interval="@once",
    catchup=False,
    params={
        "first_name": "Hongbo",
        "last_name": "Miao",
    },
) as dag:

    @task(task_id="get_name")
    def get_name(params=None) -> Dict[str, str]:
        return {
            "first_name": params["first_name"],
            "last_name": params["last_name"],
        }

    @task(task_id="transform_first_name")
    def transform_first_name(name: Dict[str, str]) -> str:
        return name["first_name"][0]

    @task
    def transform_last_name(name: Dict[str, str]) -> str:
        return name["last_name"][0]

    @task
    def get_initials(first_name_initial: str, last_name_initial: str) -> str:
        return f"{first_name_initial}.{last_name_initial}."

    get_time = BashOperator(
        task_id="get_time",
        bash_command="date '+%Y-%m-%d %H:%M:%S'",
    )

    @task
    def greet(initials: str, ti=None) -> str:
        time = ti.xcom_pull(task_ids="get_time")
        dt = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        return f'Hello {initials}, {dt.strftime("at %H:%M on %B %d, %Y")}!'

    @task_group
    def initials_group_function(name: Dict[str, str]) -> str:
        return get_initials(transform_first_name(name), transform_last_name(name))

    my_name = get_name()
    my_initials = initials_group_function(my_name)
    run_greet = greet(my_initials)

    [my_initials, get_time] >> run_greet
