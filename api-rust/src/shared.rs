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
