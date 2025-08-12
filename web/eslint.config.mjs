// https://eslint.org/docs/latest/use/configure/configuration-files

import eslint from '@eslint/js';
import tanstackQueryPlugin from '@tanstack/eslint-plugin-query';
import typescriptEslintPlugin from '@typescript-eslint/eslint-plugin';
import typescriptParser from '@typescript-eslint/parser';
import airbnb from 'eslint-config-airbnb';
import importPlugin from 'eslint-plugin-import';
import jestPlugin from 'eslint-plugin-jest';
import jestDomPlugin from 'eslint-plugin-jest-dom';
import jsxA11yPlugin from 'eslint-plugin-jsx-a11y';
import prettierPluginRecommended from 'eslint-plugin-prettier/recommended';
import reactPlugin from 'eslint-plugin-react';
import reactRefreshPlugin from 'eslint-plugin-react-refresh';
import securityPlugin from 'eslint-plugin-security';
import testingLibraryPlugin from 'eslint-plugin-testing-library';
import globals from 'globals';

/**
 * @type {Array<import('eslint').Linter.Config>}
 */
export default [
  {
    // https://eslint.org/docs/latest/use/configure/ignore#the-eslintignore-file
    ignores: [
      // Anywhere
      '**/__pycache__',
      '**/.deepeval',
      '**/.DS_Store',
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
      '**/*.aliases',
      '**/*.asv',
      '**/*.cache',
      '**/*.cf',
      '**/*.DotSettings.user',
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
      '**/build',
      '**/cmake-build-debug',
      '**/codegen',
      '**/coverage',
      '**/node_modules',
      '**/slprj',
      '**/storybook-static',
      '**/target',

      // eslint.config.mjs specific
      'src/components/ui',
      'tmp',
    ],
  },
  eslint.configs.recommended,
  {
    files: ['**/*.{cjs,mjs,ts,tsx}'],
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
        ...globals.node,
        ...globals.jest,
      },
    },
    plugins: {
      '@tanstack/query': tanstackQueryPlugin,
      '@typescript-eslint': typescriptEslintPlugin,
      import: importPlugin,
      jest: jestPlugin,
      'jest-dom': jestDomPlugin,
      'jsx-a11y': jsxA11yPlugin,
      react: reactPlugin,
      'react-refresh': reactRefreshPlugin,
      security: securityPlugin,
      'testing-library': testingLibraryPlugin,
    },
    settings: {
      'import/resolver': {
        node: {
          extensions: ['.cjs', '.mjs', '.ts', '.tsx'],
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
      ...importPlugin.configs.recommended.rules,
      ...jestPlugin.configs.recommended.rules,
      ...jestDomPlugin.configs.recommended.rules,
      ...jsxA11yPlugin.configs.recommended.rules,
      ...reactPlugin.configs.recommended.rules,
      ...reactRefreshPlugin.configs.vite.rules,
      ...tanstackQueryPlugin.configs.recommended.rules,
      ...securityPlugin.configs.recommended.rules,
      ...testingLibraryPlugin.configs.react.rules,
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
      'react-refresh/only-export-components': 'error',
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
  },
  // scripts
  {
    files: ['scripts/**/*.ts'],
    rules: {
      'import/no-unresolved': 'off',
    },
  },
  prettierPluginRecommended, // Make sure this is last
];
