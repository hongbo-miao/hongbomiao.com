import isProduction from './isProduction';

describe('isProduction', () => {
  test('return true for production', () => {
    expect(isProduction('production')).toEqual(true);
  });

  test('return false for development', () => {
    expect(isProduction('development')).toEqual(false);
  });

  test('return false for test', () => {
    expect(isProduction('test')).toEqual(false);
  });
});
