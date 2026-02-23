pub mod audio {
    pub mod utils {
        pub mod decode_flac_bytes_to_pcm_i16;
        pub mod encode_pcm_i16_to_flac_bytes;
    }
}

pub mod nats {
    pub mod services {
        pub mod process_audio_stream_from_nats;
    }
    pub mod types {
        pub mod audio_metadata;
    }
    pub mod utils {
        pub mod build_deterministic_message_id;
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
