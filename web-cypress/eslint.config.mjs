// https://eslint.org/docs/latest/use/configure/configuration-files

import eslint from '@eslint/js';
import eslintPlugin from '@typescript-eslint/eslint-plugin';
import typescriptParser from '@typescript-eslint/parser';
import airbnbBase from 'eslint-config-airbnb-base';
import cypressPlugin from 'eslint-plugin-cypress';
import importPlugin from 'eslint-plugin-import';
import prettierPluginRecommended from 'eslint-plugin-prettier/recommended';
import globals from 'globals';

/**
 * @type {Array<import('eslint').Linter.Config>}
 */
export default [
  {
    // https://eslint.org/docs/latest/use/configure/ignore#the-eslintignore-file
    ignores: [
      // Anywhere
      '**/*.DotSettings.user',
      '**/*.aliases',
      '**/*.asv',
      '**/*.cache',
      '**/*.cf',
      '**/*.ghw',
      '**/*.iml',
      '**/*.lvlps',
      '**/*.mexmaca64',
      '**/*.mexmaci64',
      '**/*.pdf',
      '**/*.slxc',
      '**/*.swc',
      '**/*.tfstate',
      '**/*.unsealed.yaml',
      '**/.DS_Store',
      '**/.ansible',
      '**/.coverage',
      '**/.deepeval',
      '**/.env.*.local',
      '**/.gitkeep',
      '**/.idea',
      '**/.pytest_cache',
      '**/.ruff_cache',
      '**/.terraform',
      '**/.vagrant',
      '**/.venv',
      '**/.vscode',
      '**/.windsurfrules',
      '**/__pycache__',
      '**/build',
      '**/cmake-build-debug',
      '**/codegen',
      '**/coverage',
      '**/coverage.xml',
      '**/node_modules',
      '**/secrets.auto.tfvars',
      '**/slprj',
      '**/target',

      // Directories
      'cypress/fixtures/example.json',
      'cypress/screenshots',
    ],
  },
  eslint.configs.recommended,
  {
    files: ['**/*.{mjs,ts}'],
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
        ...globals.mocha,
        ...globals.node,
        Cypress: 'readonly',
        cy: 'readonly',
        expect: 'readonly',
      },
    },
    plugins: {
      '@typescript-eslint': eslintPlugin,
      cypress: cypressPlugin,
      import: importPlugin,
    },
    settings: {
      'import/resolver': {
        node: {
          extensions: ['.mjs', '.ts'],
        },
        typescript: {
          alwaysTryTypes: true,
          project: 'tsconfig.json',
        },
      },
    },
    rules: {
      ...eslintPlugin.configs.recommended.rules,
      ...cypressPlugin.configs.recommended.rules,
      ...airbnbBase.rules,
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
  prettierPluginRecommended, // Make sure this is last
];
