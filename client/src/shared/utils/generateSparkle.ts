import random from './random';

const SPARKLE_COLOR = '#FFC700';

interface Sparkle {
  id: string;
  createdAt: number;
  color: string;
  size: number;
  style: {
    top: string;
    left: string;
  };
}

const generateSparkle = (): Sparkle => {
  return {
    id: String(random(10000, 99999)),
    createdAt: Date.now(),
    color: SPARKLE_COLOR,
    size: random(10, 20),
    style: {
      top: `${random(0, 100)}%`,
      left: `${random(0, 100)}%`,
    },
  };
};

export default generateSparkle;
