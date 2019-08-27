module.exports = {
  parser: '@typescript-eslint/parser',
  plugins: [
    'react',
    '@typescript-eslint',
  ],
  extends: [
    'airbnb',
    'eslint:recommended',
    'plugin:@typescript-eslint/eslint-recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react/recommended',
    'prettier/@typescript-eslint', // Uses eslint-config-prettier to disable ESLint rules from @typescript-eslint/eslint-plugin that would conflict with prettier
    'plugin:prettier/recommended', // Enables eslint-plugin-prettier and displays prettier errors as ESLint errors. Make sure this is always the last configuration in the extends array.
  ],
  settings: {
    'import/resolver': {
      'node': {
        'extensions': ['.js','.jsx','.ts','.tsx','.d.ts'],
      },
    },
  },
  env: {
    browser: true,
    jest: true,
  },
  rules: {
    'import/no-extraneous-dependencies': ['error', {
      devDependencies: true,
    }],
    'react/jsx-filename-extension': [1, {
      extensions: ['.jsx', '.tsx'],
    }],
    'spaced-comment': ['error', 'always', {
      'markers': ['/'],
    }],
  },
};
