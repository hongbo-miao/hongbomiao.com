module.exports = {
  parser: '@typescript-eslint/parser',
  env: {
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
    'prettier', // eslint-plugin-prettier
    'security', // eslint-plugin-security
  ],
  extends: [
    'eslint:recommended', // eslint
    'airbnb-base', // eslint-config-airbnb-base
    'plugin:import/errors', // eslint-plugin-import
    'plugin:import/warnings', // eslint-plugin-import
    'plugin:import/typescript', // eslint-plugin-import
    'plugin:@typescript-eslint/recommended', // @typescript-eslint/eslint-plugin
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
    'multiline-comment-style': ['error', 'starred-block'],
    'security/detect-non-literal-fs-filename': 'off',
    'spaced-comment': [
      'error',
      'always',
      {
        markers: ['/'],
      },
    ],
  },
};
