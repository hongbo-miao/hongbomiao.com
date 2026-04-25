import type { Rnnoise } from '@shiguredo/rnnoise-wasm';

export function createDenoiseNode(audioContext: AudioContext, rnnoise: Rnnoise): ScriptProcessorNode {
  const FRAME_SIZE = rnnoise.frameSize;
  const denoiseState = rnnoise.createDenoiseState();
  const node = audioContext.createScriptProcessor(4096, 1, 1);
  const inputPending: number[] = [];
  const outputPending: number[] = [];

  node.onaudioprocess = (event) => {
    const inputData = event.inputBuffer.getChannelData(0);
    const outputData = event.outputBuffer.getChannelData(0);

    for (const sample of inputData) {
      inputPending.push(sample * 32768);
    }

    while (inputPending.length >= FRAME_SIZE) {
      const audioFrame = new Float32Array(inputPending.splice(0, FRAME_SIZE));
      denoiseState.processFrame(audioFrame);
      for (const sample of audioFrame) {
        outputPending.push(sample);
      }
    }

    for (let i = 0; i < outputData.length; i++) {
      outputData[i] = (outputPending.shift() ?? 0) / 32768;
    }
  };

  (node as unknown as { _destroyDenoiseState: () => void })._destroyDenoiseState =
    () => denoiseState.destroy();

  return node;
}
