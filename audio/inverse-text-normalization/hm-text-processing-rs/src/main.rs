#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

use text_processing_rs::normalize_sentence;

fn main() {
    let normalized_text = normalize_sentence("I have twenty one apples.");
    println!("{normalized_text}");
}
