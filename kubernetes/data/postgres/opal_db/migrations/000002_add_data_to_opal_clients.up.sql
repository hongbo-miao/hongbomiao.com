insert into opal_clients (opal_client_id, config)
values
  ('hm-opal-client', '{"entries":[{"url":"postgresql://admin@postgres-service.hm-postgres:40072/opa_db","config":{"fetcher":"PostgresFetchProvider","query":"select role, allow from roles;","connection_params":{"password":"passw0rd"},"dict_key":"role"},"topics":["policy_data"],"dst_path":"roles"}]}');
