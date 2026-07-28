//! Core logic. This is the single source of truth — Swift, Python, and
//! JavaScript contain no logic of their own, they only call into the
//! functions defined here.

#[cfg(feature = "ffi")]
pub mod ffi;

/// Add two numbers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Subtract `b` from `a`.
pub fn minus(a: i32, b: i32) -> i32 {
    a - b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    fn subtracts() {
        assert_eq!(minus(5, 3), 2);
    }

    #[test]
    fn subtracts_past_zero() {
        assert_eq!(minus(3, 5), -2);
    }
}
