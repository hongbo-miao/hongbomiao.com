pub mod audio {
    pub mod types {
        pub mod speech_segment;
    }
    pub mod utils {
        pub mod convert_pcm_bytes_to_wav;
        pub mod webrtc_vad_processor;
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
