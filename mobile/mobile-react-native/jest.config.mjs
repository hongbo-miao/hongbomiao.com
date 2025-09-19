// https://jestjs.io/docs/next/configuration

/**
 * @type {import('jest').Config}
 */
export default {
  preset: 'jest-expo',
  // https://docs.expo.dev/develop/unit-testing/
  transformIgnorePatterns: [
    'node_modules/(?!((jest-)?react-native|@react-native(-community)?)|expo(nent)?|@expo(nent)?/.*|@expo-google-fonts/.*|react-navigation|@react-navigation/.*|@unimodules/.*|unimodules|sentry-expo|native-base|react-native-svg|@ui-kitten)',
  ],
  // https://github.com/expo/expo/issues/36831#issuecomment-3107047371
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
};
