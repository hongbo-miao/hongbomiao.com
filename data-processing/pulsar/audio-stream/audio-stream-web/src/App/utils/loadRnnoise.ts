import { Rnnoise } from '@shiguredo/rnnoise-wasm';

let rnnoiseInstance: Rnnoise | null = null;
let rnnoiseLoadPromise: Promise<Rnnoise> | null = null;

export function getRnnoiseInstance(): Rnnoise | null {
  return rnnoiseInstance;
}

export async function loadRnnoise(): Promise<Rnnoise> {
  if (rnnoiseInstance) return rnnoiseInstance;
  if (!rnnoiseLoadPromise) {
    rnnoiseLoadPromise = Rnnoise.load().then((instance) => {
      rnnoiseInstance = instance;
      return instance;
    }).catch((error) => {
      rnnoiseLoadPromise = null;
      throw error;
    });
  }
  return rnnoiseLoadPromise;
}
