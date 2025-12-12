// https://eslint.org/docs/latest/use/configure/configuration-files

const grafanaConfig = require("@grafana/eslint-config/flat");

/**
 * @type {Array<import('eslint').Linter.Config>}
 */
module.exports = [
  ...grafanaConfig,
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
      '**/.claude/**/*',
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
      '**/AGENTS.md',
      '**/__pycache__/**/*',
      '**/build/**/*',
      '**/certificates/**/*',
      '**/cmake-build-debug/**/*',
      '**/codegen/**/*',
      '**/coverage.xml',
      '**/coverage/**/*',
      '**/data/**/*',
      '**/node_modules/**/*',
      '**/output/**/*',
      '**/secrets.auto.tfvars',
      '**/slprj/**/*',
      '**/target/**/*',

      // Directories
      'dist',
    ],
  },
  {
    files: ['**/*.{js,ts,tsx}'],
  },
];
