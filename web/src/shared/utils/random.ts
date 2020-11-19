const random = (min: number, max: number): number => {
  return Math.floor(Math.random() * (max - min)) + min;
};

export default random;
