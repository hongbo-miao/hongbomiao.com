// https://eslint.org/docs/latest/use/configure/configuration-files

import eslint from '@eslint/js';
import typescriptEslintPlugin from '@typescript-eslint/eslint-plugin';
import typescriptParser from '@typescript-eslint/parser';
import airbnb from 'eslint-config-airbnb';
import importPlugin from 'eslint-plugin-import';
import prettierPluginRecommended from 'eslint-plugin-prettier/recommended';
import reactPlugin from 'eslint-plugin-react';
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
      '**/node_modules',
      '**/secrets.auto.tfvars',
      '**/slprj',
      '**/target',

      // Directories
      '.expo',
      'coverage',
    ],
  },
  eslint.configs.recommended,
  {
    files: ['**/*.{mjs,tsx}'],
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
        ...globals.browser,
        ...globals.jest,
      },
    },
    plugins: {
      '@typescript-eslint': typescriptEslintPlugin,
      import: importPlugin,
      react: reactPlugin,
    },
    settings: {
      'import/resolver': {
        node: {
          extensions: ['.mjs', '.tsx'],
        },
        typescript: {
          alwaysTryTypes: true,
          project: 'tsconfig.json',
        },
      },
      react: {
        version: 'detect',
      },
    },
    rules: {
      ...typescriptEslintPlugin.configs.recommended.rules,
      ...airbnb.rules,
      ...reactPlugin.configs.recommended.rules,
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
      'react/jsx-filename-extension': [
        1,
        {
          extensions: ['.tsx'],
        },
      ],
      'react/prop-types': 'off',
      'react/react-in-jsx-scope': 'off',
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
  },
  {
    files: ['**/*.{ts,tsx}'],
    rules: {
      'prettier/prettier': 'error',
    },
  },
  prettierPluginRecommended, // Make sure this is last
];
