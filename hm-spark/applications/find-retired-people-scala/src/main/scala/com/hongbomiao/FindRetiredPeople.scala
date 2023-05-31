package com.hongbomiao

import org.apache.spark.sql.{DataFrame, SparkSession}

object FindRetiredPeople {
  def main(args: Array[String]): Unit = {
    val people = Seq(
      (1, "Alice", 25),
      (2, "Bob", 30),
      (3, "Charlie", 80),
      (4, "Dave", 40),
      (5, "Eve", 45)
    )

    val spark: SparkSession = SparkSession
      .builder()
      .getOrCreate()

    import spark.implicits._
    val df: DataFrame = people.toDF("id", "name", "age")
    df.createOrReplaceTempView("people")

    val retiredPeople: DataFrame =
      spark.sql("SELECT name, age FROM people WHERE age >= 67")
    retiredPeople.show()

    spark.stop()
  }
}
