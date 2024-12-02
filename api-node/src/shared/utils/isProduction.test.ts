import NodeEnv from './NodeEnv.js';
import isProduction from './isProduction.js';

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
