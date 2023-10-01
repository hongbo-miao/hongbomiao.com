from sedona.spark import SedonaContext


def main() -> None:
    sedona_config = (
        SedonaContext.builder()
        .config(
            "spark.jars.packages",
            # https://mvnrepository.https://mvnrepository.com/artifact/org.apache.sedona
            "org.apache.sedona:sedona-spark-shaded-3.4_2.12:1.4.1,"
            # https://mvnrepository.com/artifact/org.datasyslab/geotools-wrapper
            "org.datasyslab:geotools-wrapper:1.4.0-28.2",
        )
        .getOrCreate()
    )
    sedona = SedonaContext.create(sedona_config)

    (
        sedona.read.format("csv")
        .option("delimiter", ",")
        .option("header", "false")
        # https://github.com/apache/sedona/blob/master/binder/data/testpoint.csv
        .load("data/testpoint.csv")
    ).createOrReplaceTempView("points")

    sedona.sql(
        """
        select st_point(cast(points._c0 as double), cast(points._c1 as double)) as point
        from points
        """
    ).createOrReplaceTempView("points1")
    sedona.sql(
        """
        select st_point(cast(points._c0 as double), cast(points._c1 as double)) as point
        from points
        """
    ).createOrReplaceTempView("points2")

    df = sedona.sql(
        """
        select
          points1.point as point1,
          points2.point as point2,
          st_distance(points1.point, points2.point) as distance
        from points1, points2
        where st_distance(points1.point, points2.point) < 2
        """
    )
    df.show()


if __name__ == "__main__":
    main()
