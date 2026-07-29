// JavaScript wrapper. Marshalling only — the logic lives in Rust.
//
// Loads the same C ABI, compiled to wasm32 instead of a native dylib. No
// wasm-bindgen and no npm dependencies: the exports are the plain `extern "C"`
// functions. Integers pass straight through, so nothing needs to touch the
// module's linear memory.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const WASM_PATH = fileURLToPath(
  new URL("../target/wasm32-unknown-unknown/release/calculator.wasm", import.meta.url),
);

const { instance } = await WebAssembly.instantiate(await readFile(WASM_PATH));
const wasm = instance.exports;

/** Add two numbers. */
export function add(a, b) {
  return wasm.calculator_add(a, b);
}

/** Subtract b from a. */
export function minus(a, b) {
  return wasm.calculator_minus(a, b);
}

if (import.meta.filename === process.argv[1]) {
  const a = 5;
  const b = 3;

  console.log(`${a} + ${b} = ${add(a, b)}`);
  console.log(`${a} - ${b} = ${minus(a, b)}`);
}
