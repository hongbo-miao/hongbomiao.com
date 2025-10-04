insert into opal_client (id, name, config)
values
(
    '0bdaa0c2-43fd-4f3a-b1e0-64bde83e9774',
    'hm-opal-client',
    '{"entries":[{"url":"postgresql://admin@postgres-service.hm-postgres.svc:5432/opa_db","config":{"fetcher":"PostgresFetchProvider","query":"select role, allow from role where opal_client_id = (select id from opal_client where name = ''hm-opal-client'');","connection_params":{"password":"passw0rd"},"dict_key":"role"},"topics":["policy_data"],"dst_path":"roles"}]}'
),
(
    '9b2ad6b8-555d-4d62-a644-2a96d0c0dbe5',
    'config-loader',
    null
);
