module.exports = {
  parser: '@typescript-eslint/parser',
  env: {
    browser: true,
    jest: true,
    node: true,
  },
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true,
    },
  },
  plugins: [
    '@typescript-eslint', // @typescript-eslint/eslint-plugin
    'import', // eslint-plugin-import
    'jest', // eslint-plugin-jest
    'jest-dom', // eslint-plugin-jest-dom
    'jsx-a11y', // eslint-plugin-jsx-a11y
    'prettier', // eslint-plugin-prettier
    'react', // eslint-plugin-react
    'security', // eslint-plugin-security
    'testing-library', // eslint-plugin-testing-library
  ],
  extends: [
    'eslint:recommended', // eslint
    'airbnb', // eslint-config-airbnb
    'plugin:import/errors', // eslint-plugin-import
    'plugin:import/warnings', // eslint-plugin-import
    'plugin:import/typescript', // eslint-plugin-import
    'plugin:@typescript-eslint/recommended', // @typescript-eslint/eslint-plugin
    'plugin:jest/recommended', // eslint-plugin-jest
    'plugin:jest-dom/recommended', // eslint-plugin-jest-dom
    'plugin:testing-library/react', // eslint-plugin-testing-library
    'plugin:react/recommended', // eslint-plugin-react
    'plugin:jsx-a11y/recommended', // eslint-plugin-jsx-a11y
    'plugin:security/recommended-legacy', // eslint-plugin-security
    'prettier', // eslint-plugin-prettier. Make sure to put it last in the extends array, so it gets the chance to override other configs
    'plugin:prettier/recommended', // eslint-plugin-prettier. Exposes a "recommended" configuration that configures both eslint-plugin-prettier and eslint-config-prettier in a single step. Make sure this is always the last configuration in the extends array
  ],
  settings: {
    'import/resolver': {
      node: {
        extensions: ['.js', '.ts', '.tsx'],
      },
    },
  },
  rules: {
    'import/extensions': [
      'error',
      'ignorePackages',
      {
        ts: 'never',
        tsx: 'never',
      },
    ],
    'import/no-extraneous-dependencies': [
      'error',
      {
        devDependencies: true,
      },
    ],
    'import/order': [
      'error',
      {
        // https://github.com/benmosher/eslint-plugin-import/blob/master/docs/rules/order.md#groups-array
        groups: ['builtin', 'external', 'parent', 'sibling', 'index'],
        alphabetize: {
          order: 'asc',
        },
        'newlines-between': 'never',
      },
    ],
    'jest/expect-expect': 'off',
    'multiline-comment-style': ['error', 'starred-block'],
    'react/jsx-filename-extension': [
      1,
      {
        extensions: ['.tsx'],
      },
    ],
    'react/prop-types': 'off',
    'security/detect-non-literal-fs-filename': 'off',
    'spaced-comment': [
      'error',
      'always',
      {
        markers: ['/'],
      },
    ],

    // Note must disable the base rule as it can report incorrect errors
    'no-use-before-define': 'off',
    '@typescript-eslint/no-use-before-define': ['error'],
    'no-shadow': 'off',
    '@typescript-eslint/no-shadow': 'error',
  },
};
