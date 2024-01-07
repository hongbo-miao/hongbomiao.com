with
new_data as (
    select
        date,
        item_description as item_name,
        sum(bottles_sold) as total_amount_sold
    from
        `bigquery-public-data.iowa_liquor_sales.sales`
    group by
        date,
        item_name
    having
        date between date('2021-01-01') and date('2021-12-31')
        and lower(item_name) in (
            'black velvet',
            'captain morgan spiced rum',
            'hawkeye vodka',
            "five o'clock vodka",
            'fireball cinnamon whiskey'
        )
)

select *
from
    ml.detect_anomalies(
        model `sales.hm_sales_anomalies_model`,
        struct(0.99 as anomaly_prob_threshold),
        (
            select *
            from
                new_data
        ));
