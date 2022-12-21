select *
from
    ml.detect_anomalies(
        model `hm_sales.my_arima_plus_model`,
        struct(0.8 as anomaly_prob_threshold)
    );
