// https://eslint.org/docs/latest/use/configure/configuration-files

const grafanaConfig = require("@grafana/eslint-config/flat");

/**
 * @type {Array<import('eslint').Linter.Config>}
 */
module.exports = [
  {
    // https://eslint.org/docs/latest/use/configure/ignore#the-eslintignore-file
    ignores: [
      // Anywhere
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
      '**/*.slxc',
      '**/*.swc',
      '**/*.tfstate',
      '**/*.unsealed.yaml',
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
      '**/__pycache__',
      '**/build',
      '**/cmake-build-debug',
      '**/codegen',
      '**/coverage',
      '**/node_modules',
      '**/slprj',
      '**/target',

      // Directories
      'dist',
    ],
  },
  {
    ...grafanaConfig,
  },
  {
    files: ['**/*.{js,ts,tsx}'],
  },
];
