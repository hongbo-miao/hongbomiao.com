pub mod application {
    pub mod types {
        pub mod application_state;
    }
}
pub mod audio {
    pub mod types {
        pub mod speech_segment;
    }
    pub mod utils {
        pub mod convert_pcm_bytes_to_wav;
        pub mod webrtc_vad_processor;
    }
}
pub mod database {
    pub mod types {
        pub mod pg_graphql_request;
        pub mod pg_graphql_response;
    }
    pub mod utils {
        pub mod initialize_pool;
        pub mod resolve_graphql;
    }
}
pub mod fire_audio_stream {
    pub mod constants {
        pub mod fire_streams;
    }
    pub mod utils {
        pub mod consume_fire_audio_stream_from_nats;
        pub mod fire_audio_stream_manager;
        pub mod fire_stream_state;
        pub mod handle_fire_audio_websocket;
        pub mod process_fire_audio_stream;
    }
}
pub mod image {
    pub mod utils {
        pub mod load_labels;
        pub mod load_model;
        pub mod process_image;
    }
}
pub mod openai {
    pub mod types {
        pub mod chat_response;
    }
    pub mod utils {
        pub mod chat;
    }
}
pub mod parallel_calculation {
    pub mod types {
        pub mod calculation_response;
    }
    pub mod utils {
        pub mod calculate_parallel;
    }
}
pub mod python_parallel_calculation {
    pub mod types {
        pub mod python_calculation_response;
    }
    pub mod utils {
        pub mod calculate_with_python;
    }
}
pub mod police_audio_stream {
    pub mod constants {
        pub mod police_streams;
    }
    pub mod utils {
        pub mod handle_police_audio_websocket;
        pub mod police_audio_stream_manager;
        pub mod police_stream_state;
        pub mod process_police_audio_stream;
        pub mod spawn_transcription_and_broadcast;
    }
}
pub mod speaches {
    pub mod services {
        pub mod transcribe_audio;
    }
}
pub mod server_sent_event {
    pub mod types {
        pub mod server_sent_event_message;
        pub mod server_sent_event_query;
    }
    pub mod utils {
        pub mod broadcast_transcription_result;
        pub mod server_sent_event_manager;
    }
}
