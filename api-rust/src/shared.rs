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
        pub mod process_police_audio_stream;
    }
}
