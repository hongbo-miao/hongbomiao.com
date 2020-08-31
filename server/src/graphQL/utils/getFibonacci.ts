import validator from 'validator';
import calcFibonacci from './calcFibonacci';

interface Fibonacci {
  n: number;
  ans: number;
}

const getFibonacci = (n: number): Fibonacci => {
  if (!validator.isInt(String(n), { min: 0, max: 10 })) {
    throw new Error('n should be in the range of 0 and 10.');
  }
  return {
    n,
    ans: calcFibonacci(n),
  };
};

export default getFibonacci;
