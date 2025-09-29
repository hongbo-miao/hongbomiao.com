// https://eslint.org/docs/latest/use/configure/configuration-files

import eslint from '@eslint/js';
import eslintPlugin from '@typescript-eslint/eslint-plugin';
import typescriptParser from '@typescript-eslint/parser';
import airbnbBase from 'eslint-config-airbnb-base';
import importPlugin from 'eslint-plugin-import';
import prettierPluginRecommended from 'eslint-plugin-prettier/recommended';
import securityPlugin from 'eslint-plugin-security';
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
      '**/*.swc/**/*',
      '**/*.tfstate',
      '**/*.unsealed.yaml',
      '**/.DS_Store',
      '**/.ansible/**/*',
      '**/.coverage/**/*',
      '**/.deepeval/**/*',
      '**/.env.*.local',
      '**/.gitkeep',
      '**/.idea/**/*',
      '**/.pytest_cache/**/*',
      '**/.ruff_cache/**/*',
      '**/.terraform/**/*',
      '**/.vagrant/**/*',
      '**/.venv/**/*',
      '**/.vscode/**/*',
      '**/.windsurfrules',
      '**/__pycache__/**/*',
      '**/build/**/*',
      '**/cmake-build-debug/**/*',
      '**/codegen/**/*',
      '**/coverage.xml',
      '**/coverage/**/*',
      '**/data/**/*',
      '**/node_modules/**/*',
      '**/secrets.auto.tfvars',
      '**/slprj/**/*',
      '**/target/**/*',

      // Directories
      'build',
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
        ...globals.node,
      },
    },
    plugins: {
      '@typescript-eslint': eslintPlugin,
      import: importPlugin,
      security: securityPlugin,
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
      ...airbnbBase.rules,
      ...securityPlugin.configs.recommended.rules,
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
  },
  prettierPluginRecommended, // Make sure this is last
];
