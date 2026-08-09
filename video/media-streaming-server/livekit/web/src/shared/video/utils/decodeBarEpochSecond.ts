// The publisher draws a single bright bar whose slot encodes the wall-clock second the
// frame was generated at, modulo BAR_STEP_COUNT. Downsampling the frame to one pixel per
// slot and picking the brightest slot recovers that second. The step only carries
// whole-second resolution, so the returned epoch second is reconstructed by picking the
// candidate closest to now, and the caller derives sub-second latency by comparing it
// against a continuously advancing local clock rather than by refining the decode itself.
const BAR_STEP_COUNT = 20;
const MINIMUM_BRIGHTNESS = 96;

export default function decodeBarEpochSecond(
  videoElement: HTMLVideoElement,
  canvasElement: HTMLCanvasElement,
): number | null {
  if (videoElement.videoWidth === 0 || videoElement.readyState < 2) {
    return null;
  }

  const context = canvasElement.getContext('2d', { willReadFrequently: true });
  if (context == null) {
    return null;
  }

  canvasElement.width = BAR_STEP_COUNT;
  canvasElement.height = 1;
  context.drawImage(videoElement, 0, 0, BAR_STEP_COUNT, 1);
  const { data } = context.getImageData(0, 0, BAR_STEP_COUNT, 1);

  let brightestStep = -1;
  let brightestValue = MINIMUM_BRIGHTNESS;
  for (let step = 0; step < BAR_STEP_COUNT; step += 1) {
    const brightness = data[step * 4]!;
    if (brightness > brightestValue) {
      brightestValue = brightness;
      brightestStep = step;
    }
  }

  if (brightestStep < 0) {
    return null;
  }

  const localEpochSecond = Math.round(Date.now() / 1_000);
  const secondsSinceStep = (((localEpochSecond % BAR_STEP_COUNT) - brightestStep) + BAR_STEP_COUNT) % BAR_STEP_COUNT;
  return localEpochSecond - secondsSinceStep;
}
