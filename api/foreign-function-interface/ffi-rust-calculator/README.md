# FFI Rust Calculator

An FFI (Foreign Function Interface) demo: write the logic once in Rust, expose it through a C ABI (Application Binary Interface), and let every other language be a thin wrapper.
Here Rust, Swift, Python, and JavaScript all call the same `add` and `minus` — none of them reimplements anything.

An ABI is the machine-level contract between compiled code: how arguments are passed in registers, how the stack is laid out, how symbols are named.
The C one is what every OS and toolchain already agrees on, which is why it works as the shared boundary.
