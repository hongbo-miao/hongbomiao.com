from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode


def main(data_dirname: str, ratings_filename: str, movies_filename: str) -> None:
    spark = SparkSession.builder.getOrCreate()

    ratings_df = spark.read.csv(
        f"{data_dirname}/{ratings_filename}",
        header=True,
        inferSchema=True,
    )
    movies_df = spark.read.csv(
        f"{data_dirname}/{movies_filename}",
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
        ratings_df.select("userId").distinct(),
        top_movie_count,
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
    # https://grouplens.org/datasets/movielens/latest/
    external_data_dirname = "data"
    external_ratings_filename = "ratings.csv"
    external_movies_filename = "movies.csv"
    main(external_data_dirname, external_ratings_filename, external_movies_filename)
