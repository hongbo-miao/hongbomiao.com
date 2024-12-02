import validator from 'validator';
import calcFibonacci from './calcFibonacci.js';

type Fibonacci = {
  n: number;
  ans: number;
};

const getFibonacci = (n: number): Fibonacci => {
  if (!validator.isInt(String(n), { min: 0, max: 20 })) {
    throw new Error('n should be in the range of 0 and 20.');
  }
  return {
    n,
    ans: calcFibonacci(n),
  };
};

export default getFibonacci;
