import CCalculator

/// Swift wrapper. Marshalling only — the logic lives in Rust.
public enum Calculator {
    /// Add two numbers.
    public static func add(_ a: Int32, _ b: Int32) -> Int32 {
        calculator_add(a, b)
    }

    /// Subtract `b` from `a`.
    public static func minus(_ a: Int32, _ b: Int32) -> Int32 {
        calculator_minus(a, b)
    }
}

// Demo entry point.
let a: Int32 = 5
let b: Int32 = 3

print("\(a) + \(b) = \(Calculator.add(a, b))")
print("\(a) - \(b) = \(Calculator.minus(a, b))")
