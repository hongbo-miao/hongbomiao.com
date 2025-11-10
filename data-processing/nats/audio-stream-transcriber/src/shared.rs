pub mod audio {
    pub mod utils {
        pub mod encode_pcm_i16_to_flac_bytes;
    }
}

pub mod nats {
    pub mod services {
        pub mod process_audio_stream_from_nats;
    }
    pub mod utils {
        pub mod publish_transcription;
    }
}

pub mod speaches {
    pub mod services {
        pub mod transcribe_audio;
    }
    pub mod types {
        pub mod transcription_response;
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
