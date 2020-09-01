module.exports = {
  root: true,
  parser: '@typescript-eslint/parser',
  env: {
    jest: true,
    node: true,
  },
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true,
    },
  },
  plugins: [
    '@babel', // @babel/eslint-plugin
    '@typescript-eslint', // @typescript-eslint/eslint-plugin
    'jest', // eslint-plugin-jest
    'prettier', // eslint-plugin-prettier
    'security', // eslint-plugin-security
  ],
  extends: [
    'eslint:recommended', // eslint
    'airbnb-base', // eslint-config-airbnb-base
    'plugin:@typescript-eslint/eslint-recommended', // @typescript-eslint/eslint-plugin. Not all eslint core rules are compatible with TypeScript, so you need to add both eslint:recommended and plugin:@typescript-eslint/eslint-recommended
    'plugin:@typescript-eslint/recommended', // @typescript-eslint/eslint-plugin
    'plugin:jest/recommended', // eslint-plugin-jest
    'plugin:security/recommended', // eslint-plugin-security
    'prettier', // eslint-plugin-prettier. Make sure to put it last in the extends array, so it gets the chance to override other configs
    'prettier/@typescript-eslint', // eslint-config-prettier. Disables ESLint rules from @typescript-eslint/eslint-plugin that would conflict with prettier
    'prettier/babel', // eslint-config-prettier
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
