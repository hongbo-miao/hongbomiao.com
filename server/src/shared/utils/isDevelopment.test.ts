import isDevelopment from './isDevelopment';

describe('isDevelopment', () => {
  test('return true for development', () => {
    expect(isDevelopment('development')).toEqual(true);
  });

  test('return false for production', () => {
    expect(isDevelopment('production')).toEqual(false);
  });

  test('return false for test', () => {
    expect(isDevelopment('test')).toEqual(false);
  });
});
