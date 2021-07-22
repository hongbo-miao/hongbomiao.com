import NodeEnv from './NodeEnv';
import isDevelopment from './isDevelopment';

describe('isDevelopment', () => {
  test('return true for development', () => {
    expect(isDevelopment(NodeEnv.development)).toEqual(true);
  });

  test('return false for production', () => {
    expect(isDevelopment(NodeEnv.production)).toEqual(false);
  });

  test('return false for test', () => {
    expect(isDevelopment(NodeEnv.test)).toEqual(false);
  });
});
