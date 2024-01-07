select sqrt(mean_squared_error) as rmse
from
    ml.evaluate(model `taxi.hm_taxi_fare_model`, (
        with
        params as (
            select
                1 as train,
                2 as eval
        ),

        daynames as (
            select [
                'sun',
                'mon',
                'tues',
                'wed',
                'thurs',
                'fri',
                'sat'
            ] as daysofweek
        ),

        taxitrips as (
            select
                pickup_longitude as pickuplon,
                pickup_latitude as pickuplat,
                dropoff_longitude as dropofflon,
                dropoff_latitude as dropofflat,
                passenger_count as passengers,
                (tolls_amount + fare_amount) as total_fare,
                daysofweek[ordinal(extract(dayofweek from pickup_datetime))] as dayofweek,
                extract(hour from pickup_datetime) as hourofday
            from
                `nyc-tlc.yellow.trips`,
                daynames,
                params
            where
                trip_distance > 0
                and fare_amount > 0
                and mod(abs(farm_fingerprint(cast(pickup_datetime as string))), 1000) = params.eval
        )

        select *
        from
            taxitrips
    ));
