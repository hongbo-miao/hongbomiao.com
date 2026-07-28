//! Rust consumer — links the crate natively, no FFI involved.

fn main() {
    let (a, b) = (5, 3);

    println!("{a} + {b} = {}", calculator::add(a, b));
    println!("{a} - {b} = {}", calculator::minus(a, b));
}
