import NodeEnv from '@/shared/utils/NodeEnv';
import isDevelopment from '@/shared/utils/isDevelopment';

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
