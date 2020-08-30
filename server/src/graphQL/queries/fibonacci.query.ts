import validator from 'validator';
import calFibonacci from '../../shared/utils/calFibonacci';

interface Fibonacci {
  n: number;
  ans: number;
}

const fibonacci = async (n: number): Promise<Fibonacci> => {
  if (!validator.isInt(String(n), { min: 0, max: 10 })) {
    throw new Error('n should be in the range of 0 and 10.');
  }
  return {
    n,
    ans: calFibonacci(n),
  };
};

export default fibonacci;
