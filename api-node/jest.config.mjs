// https://jestjs.io/docs/next/configuration

/**
 * @type {import('jest').Config}
 */
export default {
  testEnvironment: 'node',
  testPathIgnorePatterns: ['<rootDir>/build'],
  extensionsToTreatAsEsm: ['.ts'],
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },
  transform: {
    '^.+\\.(t|j)s?$': ['@swc/jest'],
  },
};
