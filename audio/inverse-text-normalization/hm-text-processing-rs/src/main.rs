#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

use text_processing_rs::{
    NormalizeOptions, normalize_sentence_with_options, tn_normalize_sentence,
};

fn main() {
    // Inverse text normalization (ITN)
    let options = NormalizeOptions::new()
        .with_concat_compound_numbers(true)
        .with_disable_bare_second(true);
    let result = normalize_sentence_with_options(
        "United seven eighty eight, please come up on frequency one three five point six two five. Give me a second while I finish coordinating.",
        options,
    );
    println!("{result}");

    // Text normalization (TN)
    let result = tn_normalize_sentence("I paid $5 for 23 items.");
    println!("{result}");
}
