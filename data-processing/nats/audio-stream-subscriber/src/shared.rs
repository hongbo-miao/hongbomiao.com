pub mod audio {
    pub mod utils {
        pub mod convert_pcm_bytes_to_wav;
    }
}

pub mod nats {
    pub mod services {
        pub mod audio_stream_processor;
    }
}

pub mod speaches {
    pub mod services {
        pub mod transcription_service;
    }
}

pub mod webrtc_vad {
    pub mod types {
        pub mod speech_segment;
        pub mod webrtc_vad_process_result;
    }

    pub mod states {
        pub mod speech_state;
    }

    pub mod services {
        pub mod webrtc_vad_processor;
    }
}
