create or replace model `sales.hm_sales_anomalies_model`
options(
    model_type = 'arima_plus',
    time_series_timestamp_col = 'date',
    time_series_data_col = 'total_amount_sold',
    time_series_id_col = 'item_name',
    holiday_region = 'US'
) as (
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
        date between date('2016-01-01') and date('2020-12-31')
        and lower(item_name) in (
            'black velvet',
            'captain morgan spiced rum',
            'hawkeye vodka',
            "five o'clock vodka",
            'fireball cinnamon whiskey'
        )
);
