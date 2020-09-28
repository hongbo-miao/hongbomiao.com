module.exports = {
  apps: [
    {
      name: 'server',
      script: './build/index.js',
      instances: 2, // https://pm2.keymetrics.io/docs/usage/cluster-mode/#usage
      exec_mode: 'cluster',
      args: '--showHTTPLog',
      env_development: {
        NODE_ENV: 'development',
      },
      env_production: {
        NODE_ENV: 'production',
      },
    },
  ],
};
