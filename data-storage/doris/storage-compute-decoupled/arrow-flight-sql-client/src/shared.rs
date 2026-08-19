pub mod doris {
    pub mod services {
        pub mod connect_to_flight_sql_service;
        pub mod fetch_record_batches_from_backend;
        pub mod run_sql_statements;
    }
    pub mod utils {
        pub mod extract_backend_pod_name;
        pub mod rewrite_location_uri_scheme;
        pub mod split_sql_statements;
    }
}
