#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

use text_processing_rs::{normalize_sentence, tn_normalize_sentence};

fn main() {
    // Inverse text normalization (ITN)
    let result = normalize_sentence("I have twenty one apples.");
    println!("{result}");

    // Text normalization (TN)
    let result = tn_normalize_sentence("I paid $5 for 23 items.");
    println!("{result}");
}
