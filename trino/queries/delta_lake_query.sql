select
    timestamp,
    current,
    voltage,
    temperature
from delta.hm_iot_db.motor limit 100;
