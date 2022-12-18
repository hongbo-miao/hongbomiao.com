SELECT
  *
FROM
  ML.PREDICT(MODEL `taxi.taxifare_model`, (
    WITH
      params AS (
        SELECT
          1 AS TRAIN,
          2 AS EVAL
      ),
      daynames AS (
        SELECT
          ['Sun',
          'Mon',
          'Tues',
          'Wed',
          'Thurs',
          'Fri',
          'Sat'] AS daysofweek
      ),
      taxitrips AS (
        SELECT
          (tolls_amount + fare_amount) AS total_fare,
          daysofweek[ORDINAL(EXTRACT(DAYOFWEEK FROM pickup_datetime))] AS dayofweek,
          EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
          pickup_longitude AS pickuplon,
          pickup_latitude AS pickuplat,
          dropoff_longitude AS dropofflon,
          dropoff_latitude AS dropofflat,
          passenger_count AS passengers
        FROM
          `nyc-tlc.yellow.trips`,
          daynames,
          params
        WHERE
          trip_distance > 0
          AND fare_amount > 0
          AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))),1000) = params.EVAL
      )
    SELECT
      *
    FROM
      taxitrips
  ));
