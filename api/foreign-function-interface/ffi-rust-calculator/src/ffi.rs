//! C ABI layer. Thin marshalling only — no logic lives here.

use std::ffi::c_int;

/// Add two numbers.
#[unsafe(no_mangle)]
pub extern "C" fn calculator_add(a: c_int, b: c_int) -> c_int {
    crate::add(a, b)
}

/// Subtract `b` from `a`.
#[unsafe(no_mangle)]
pub extern "C" fn calculator_minus(a: c_int, b: c_int) -> c_int {
    crate::minus(a, b)
}
