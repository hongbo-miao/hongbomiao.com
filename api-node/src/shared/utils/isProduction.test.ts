import NodeEnv from './NodeEnv';
import isProduction from './isProduction';

describe('isProduction', () => {
  test('return true for production', () => {
    expect(isProduction(NodeEnv.production)).toEqual(true);
  });

  test('return false for development', () => {
    expect(isProduction(NodeEnv.development)).toEqual(false);
  });

  test('return false for test', () => {
    expect(isProduction(NodeEnv.test)).toEqual(false);
  });
});
