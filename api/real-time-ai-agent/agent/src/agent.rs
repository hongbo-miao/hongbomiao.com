pub mod constants {
    pub mod agent_model_id;
}
pub mod handlers {
    pub mod handle_chat_completions_request;
    pub mod handle_list_models_request;
}
pub mod services {
    pub mod sensor_context_hook;
    pub mod serve_agent_endpoints;
}
pub mod tools {
    pub mod lookup_sensor_status;
    pub mod scan_recent_readings;
}
pub mod types {
    pub mod chat_completion_choice;
    pub mod chat_completion_request;
    pub mod chat_completion_response;
    pub mod chat_message;
    pub mod model_info;
    pub mod model_list_response;
}
