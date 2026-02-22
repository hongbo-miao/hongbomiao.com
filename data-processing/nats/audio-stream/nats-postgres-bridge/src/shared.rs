pub mod nats {
    pub mod services {
        pub mod subscribe_transcriptions_from_nats;
    }
    pub mod utils {
        pub mod process_message;
    }
}

pub mod postgres {
    pub mod services {
        pub mod write_transcription_to_postgres;
    }
}

pub mod transcripts {
    pub mod types {
        pub mod transcript;
    }
}
