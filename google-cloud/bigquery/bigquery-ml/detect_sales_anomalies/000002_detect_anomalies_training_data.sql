select *
from
    ml.detect_anomalies(
        model `sales.hm_sales_anomalies_model`,
        struct(0.99 as anomaly_prob_threshold)
    );
