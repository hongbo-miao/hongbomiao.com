module.exports = {
  ci: {
    collect: {
      staticDistDir: './build',
    },
    assert: {
      preset: 'lighthouse:recommended',
      assertions: {
        'csp-xss': ['warn', { minScore: 0 }],
        'errors-in-console': ['warn', { minScore: 0 }],
        'no-unload-listeners': ['warn', { minScore: 0 }],
        'offline-start-url': ['warn', { minScore: 0 }], // https://github.com/Hongbo-Miao/hongbomiao.com/issues/824
        'service-worker': ['warn', { minScore: 0 }],
        'unused-css-rules': ['warn', { minScore: 0 }],
        'unused-javascript': ['warn', { minScore: 0 }],
        'uses-rel-preconnect': ['warn', { minScore: 0 }],
        'valid-source-maps': ['warn', { minScore: 0 }],
        'works-offline': ['warn', { minScore: 0 }],
      },
    },
    upload: {
      target: 'temporary-public-storage',
    },
  },
};
