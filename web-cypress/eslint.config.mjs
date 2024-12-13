// https://eslint.org/docs/latest/use/configure/configuration-files

import eslint from '@eslint/js';
import eslintPlugin from '@typescript-eslint/eslint-plugin';
import typescriptParser from '@typescript-eslint/parser';
import airbnbBase from 'eslint-config-airbnb-base';
import prettierConfig from 'eslint-config-prettier';
import cypressPlugin from 'eslint-plugin-cypress';
import importPlugin from 'eslint-plugin-import';
import prettierPlugin from 'eslint-plugin-prettier';
import globals from 'globals';

/**
 * @type {Array<import('eslint').Linter.Config>}
 */
export default [
  {
    // https://eslint.org/docs/latest/use/configure/ignore#the-eslintignore-file
    ignores: [
      // Anywhere
      '**/**.aliases',
      '**/**.asv',
      '**/**.cache',
      '**/**.cf',
      '**/**.DotSettings.user',
      '**/**.ghw',
      '**/**.iml',
      '**/**.lvlps',
      '**/**.mexmaca64',
      '**/**.mexmaci64',
      '**/**.slxc',
      '**/**.swc',
      '**/**.tfstate',
      '**/**.unsealed.yaml',
      '**/.DS_Store',
      '**/*.duckdb',
      '**/.env.development.local',
      '**/.env.production.local',
      '**/.gitkeep',
      '**/.idea',
      '**/.pytest_cache',
      '**/.ruff_cache',
      '**/.terraform',
      '**/.vagrant',
      '**/.venv',
      '**/.vscode',
      '**/__pycache__',
      '**/build',
      '**/cmake-build-debug',
      '**/codegen',
      '**/coverage',
      '**/node_modules',
      '**/slprj',
      '**/target',

      // Directories
      'cypress/fixtures/example.json',
      'cypress/screenshots',
    ],
  },
  eslint.configs.recommended,
  {
    files: ['**/*.{js,ts}'],
    languageOptions: {
      parser: typescriptParser,
      parserOptions: {
        ecmaVersion: 2024,
        sourceType: 'module',
        ecmaFeatures: {
          jsx: true,
        },
      },
      globals: {
        ...globals.node,
        ...globals.mocha,
        cy: 'readonly',
        Cypress: 'readonly',
        expect: 'readonly',
      },
    },
    plugins: {
      '@typescript-eslint': eslintPlugin,
      cypress: cypressPlugin,
      import: importPlugin,
      prettier: prettierPlugin,
    },
    settings: {
      'import/resolver': {
        node: {
          extensions: ['.js', '.ts'],
        },
      },
    },
    rules: {
      ...eslintPlugin.configs.recommended.rules,
      ...cypressPlugin.configs.recommended.rules,
      ...airbnbBase.rules,
      ...prettierConfig.rules,
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
      'spaced-comment': [
        'error',
        'always',
        {
          markers: ['/'],
        },
      ],
    },
  },
  prettierConfig, // Make sure prettierConfig is last
];
