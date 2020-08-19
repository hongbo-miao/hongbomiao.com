module.exports = {
  ci: {
    collect: {
      staticDistDir: './build',
    },
    assert: {
      preset: 'lighthouse:recommended',
      assertions: {
        'errors-in-console': ['warn', { minScore: 0 }],
        'unused-css-rules': ['warn', { minScore: 0 }],
        'unused-javascript': ['warn', { minScore: 0 }],
        'uses-rel-preconnect': ['warn', { minScore: 0 }],
      },
    },
  },
};
