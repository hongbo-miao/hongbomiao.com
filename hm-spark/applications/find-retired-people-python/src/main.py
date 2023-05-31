from pyspark.sql import SparkSession


def main() -> None:
    people = [
        {"id": 1, "name": "Alice", "age": 25},
        {"id": 2, "name": "Bob", "age": 30},
        {"id": 3, "name": "Charlie", "age": 80},
        {"id": 4, "name": "Dave", "age": 40},
        {"id": 5, "name": "Eve", "age": 45},
    ]

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(people)
    df.createOrReplaceTempView("people")
    retired_people = spark.sql("SELECT name, age FROM people WHERE age >= 67")
    retired_people.show()
    spark.stop()


if __name__ == "__main__":
    main()
