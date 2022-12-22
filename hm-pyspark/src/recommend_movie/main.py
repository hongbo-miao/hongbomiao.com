from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode


def main():
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("recommend_movie")
        .config("spark.ui.port", "4040")
        .getOrCreate()
    )

    # https://grouplens.org/datasets/movielens/latest/
    data_dirname = "data"
    ratings_df = spark.read.csv(
        f"{data_dirname}/ratings.csv",
        header=True,
        inferSchema=True,
    )
    movies_df = spark.read.csv(
        f"{data_dirname}/movies.csv",
        header=True,
        inferSchema=True,
    )

    training, test = ratings_df.randomSplit([0.8, 0.2])

    als = ALS(
        maxIter=5,
        regParam=0.01,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
    )
    model = als.fit(training)

    user_ids = [1, 2, 3]
    top_movie_count = 5

    recommendations = model.recommendForUserSubset(
        ratings_df.select("userId").distinct(), top_movie_count
    )

    recommendations_exploded = (
        recommendations.filter(recommendations["userId"].isin(user_ids))
        .withColumn("recommendation", explode("recommendations"))
        .select(
            "userId",
            col("recommendation.movieId").alias("movieId"),
            col("recommendation.rating").alias("rating"),
        )
        .withColumnRenamed("movieId", "recommendation_movieId")
    )

    recommendations_with_titles = recommendations_exploded.join(
        movies_df,
        on=[recommendations_exploded["recommendation_movieId"] == movies_df["movieId"]],
        how="left",
    ).drop("recommendation_movieId")
    recommendations_with_titles.show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
