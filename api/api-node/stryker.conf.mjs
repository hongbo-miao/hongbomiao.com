// https://stryker-mutator.io/docs/stryker-js/config-file

/**
 * @type {import('@stryker-mutator/api/core').PartialStrykerOptions}
 */
export default {
  packageManager: 'npm',
  reporters: ['html', 'clear-text', 'progress', 'dashboard'],
  testRunner: 'jest',
  coverageAnalysis: 'perTest',
  jest: {
    configFile: 'jest.config.mjs',
    enableFindRelatedTests: true,
  },
};
