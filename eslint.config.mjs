// https://eslint.org/docs/latest/use/configure/configuration-files

import htmlPlugin from '@html-eslint/eslint-plugin';
import htmlParser from "@html-eslint/parser";
import jsonPlugin from "@eslint/json";

/**
 * @type {Array<import('eslint').Linter.Config>}
 */
export default [
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
      '**/storybook-static',
      '**/target',

      // Root (Cannot have leading /)
      '.aider.chat.history.md',
      '.aider.tags.cache.v3',
      '.git',
      '.github',
      '.mypy_cache',
      'build-desktop-qt-Qt_6_4_1_for_macOS-Debug',
      'dump.rdb',
      'logs',
      'logs.log',
      'submodules',
      'vendor',
      'west-master-k3s.yaml',

      // Directories
      'aerospace/hm-openaerostruct/reports',
      'aerospace/hm-openaerostruct/n2.html',
      'api-go/config/config_loader/server.crt',
      'api-go/config/config_loader/server.key',
      'api-go/config/config_loader/opal_auth_public_key.pem',
      'api-go/coverage.txt',
      'api-node/.clinic',
      'api-node/.stryker-tmp',
      'api-node/public',
      'api-node/reports',
      'api-python/dist',
      'api-rust/models',
      'caddy/public',
      'cloud-cost/komiser/komiser.db',
      'cloud-infrastructure/hm-pulumi/passphrase.txt',
      'cloud-infrastructure/terraform/environments/production/aws/data/files/amazon-msk/*/plugins/*.zip',
      'cloud-platform/aws/aws-secrets-manager/secrets/*-credentials.json',
      'cloud-security/hm-prowler/output',
      'compiler-infrastructure/llvm-ir/output',
      'computational-fluid-dynamics/openfoam/simulations/*/0.*',
      'computational-fluid-dynamics/openfoam/simulations/*/constant/polyMesh',
      'computer-vision/hm-imagebind/.checkpoints',
      'computer-vision/hm-imagebind/data',
      'computer-vision/hm-supervision/*/data',
      'computer-vision/open-clip/data',
      'data-analytics/hm-geopandas/data',
      'data-ingestion/fluent-bit/*/data',
      'data-ingestion/vector/*/data',
      'data-orchestration/hm-prefect/workflows/*/*-deployment.yaml',
      'data-orchestration/hm-prefect/workflows/*/.coverage',
      'data-orchestration/hm-prefect/workflows/*/coverage.xml',
      'data-processing/flink/applications/*/.classpath',
      'data-processing/flink/applications/*/.project',
      'data-processing/flink/applications/*/.settings',
      'data-processing/flink/applications/*/dependency-reduced-pom.xml',
      'data-processing/flink/applications/*/src/main/resources/*.properties',
      'data-processing/hm-spark/applications/*/.bsp',
      'data-processing/hm-spark/applications/*/data',
      'data-storage/hm-duckdb/*/data',
      'data-storage/hm-keydb/dump.rdb',
      'data-storage/hm-keydb/modules',
      'data-storage/hm-protobuf/data',
      'data-transformation/dbt/projects/*/dbt_packages',
      'data-transformation/dbt/projects/*/logs',
      'data-visualization/grafana/hm-panel-plugin/.config',
      'data-visualization/metabase/plugins',
      'desktop-qt/CMakeLists.txt.user',
      'digital-design/verilog/output',
      'embedded-systems/decode-can-data/data',
      'hardware-in-the-loop/national-instruments/hm-tdms/data',
      'hardware-in-the-loop/national-instruments/veristand/VeriStandZeroMQBridge/packages',
      'hardware-in-the-loop/national-instruments/veristand/VeriStandZeroMQBridge/VeriStandZeroMQBridge/bin',
      'hardware-in-the-loop/national-instruments/veristand/VeriStandZeroMQBridge/VeriStandZeroMQBridge/obj',
      'hm-kafka/kafka-client/kafka-c/*/config.ini',
      'kubernetes/certificates',
      'kubernetes/data/config-loader',
      'kubernetes/data/elastic-apm',
      'kubernetes/data/hasura/hasura-graphql-engine',
      'kubernetes/data/hm-alpine',
      'kubernetes/data/hm-kafka/hm-kafka',
      'kubernetes/data/hm-kafka/logging-kafka-connect',
      'kubernetes/data/hm-kafka/opa-kafka-connect',
      'kubernetes/data/minio',
      'kubernetes/data/model-server/model-store',
      'kubernetes/data/opal-server',
      'kubernetes/data/yugabyte',
      'kubernetes/manifests-raw',
      'machine-learning/convolutional-neural-network/output/models',
      'machine-learning/convolutional-neural-network/output/reports',
      'machine-learning/convolutional-neural-network/wandb',
      'machine-learning/feature-store/driver_features/data',
      'machine-learning/graph-neural-network/dataset',
      'machine-learning/graph-neural-network/wandb',
      'machine-learning/hm-autogluon/AutogluonModels',
      'machine-learning/hm-docling/data',
      'machine-learning/hm-faster-whisper/data',
      'machine-learning/hm-faster-whisper/output',
      'machine-learning/hm-langchain/applications/*/data',
      'machine-learning/hm-langgraph/applications/*/data',
      'machine-learning/hm-llama-index/applications/*/data',
      'machine-learning/hm-mlflow/experiments/*/data',
      'machine-learning/hm-mlflow/experiments/*/lightning_logs',
      'machine-learning/hm-mlflow/experiments/*/mlruns',
      'machine-learning/hm-mlflow/experiments/*/wandb',
      'machine-learning/hm-rasa/.rasa',
      'machine-learning/hm-rasa/graph.html',
      'machine-learning/hm-rasa/models',
      'machine-learning/hm-rasa/results',
      'machine-learning/hm-rasa/story_graph.dot',
      'machine-learning/mineru/data',
      'machine-learning/mineru/output',
      'machine-learning/neural-forecasting/*/lightning_logs',
      'machine-learning/stable-diffusion/output',
      'machine-learning/triton-inference-server/amazon-sagemaker-triton-resnet-50/infer/data',
      'mobile/mobile-android/.gradle',
      'mobile/mobile-android/local.properties',
      'mobile/mobile-ios/HMMobile.xcodeproj/project.xcworkspace',
      'mobile/mobile-ios/HMMobile.xcodeproj/xcuserdata',
      'mobile/mobile-react-native/.expo',
      'reverse-engineering/*/main',
      'robotics/robot-operating-system/bags',
      'robotics/robot-operating-system/install',
      'robotics/robot-operating-system/log',
      'web-cypress/cypress/fixtures/example.json',
      'web-cypress/cypress/screenshots',
      'web/.eslintcache',
      'web/.lighthouseci',
      'web/public/sitemap.xml',
      'web/storybook-static',
      'web/tmp',

      // eslint.config.mjs specific
      '**/*.mjs',
      '**/package-lock.json',
      '**/tsconfig.json',
    ],
  },
  {
    files: ["**/*.html"],
    languageOptions: {
      parser: htmlParser,
    },
    plugins: {
      '@html-eslint': htmlPlugin,
    },
    rules: {
      ...htmlPlugin.configs["flat/recommended"].rules,
      "@html-eslint/indent": ["error", 2],
    },
  },
  {
    files: ["**/*.json"],
    language: "json/json",
    plugins: {
      json: jsonPlugin,
    },
    rules: {
      ...jsonPlugin.configs.recommended.rules,
    },
  },
  {
    files: ["**/*.jsonc"],
    language: "json/jsonc",
    plugins: {
      json: jsonPlugin,
    },
    rules: {
      ...jsonPlugin.configs.recommended.rules,
    },
  },
  {
    files: ["**/*.json5"],
    language: "json/json5",
    plugins: {
      json: jsonPlugin,
    },
    rules: {
      ...jsonPlugin.configs.recommended.rules,
    },
  },
];
