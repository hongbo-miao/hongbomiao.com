conn = database('hm-trino', 'hadoop', '');

query = 'select * from delta.hm_iot_db.motor limit 100';
data = fetch(conn, query);

close(conn);
clear conn;
clear query;
