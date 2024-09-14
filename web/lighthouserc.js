module.exports = {
  ci: {
    collect: {
      staticDistDir: './build',
    },
    assert: {
      preset: 'lighthouse:recommended',
      assertions: {
        'bf-cache': ['warn', { minScore: 0 }],
        'color-contrast': ['warn', { minScore: 0 }],
        'csp-xss': ['warn', { minScore: 0 }],
        deprecations: ['warn', { minScore: 0 }],
        'errors-in-console': ['warn', { minScore: 0 }],
        'inspector-issues': ['warn', { minScore: 0 }],
        'landmark-one-main': ['warn', { minScore: 0 }],
        'no-unload-listeners': ['warn', { minScore: 0 }],
        'offline-start-url': ['warn', { minScore: 0 }], // https://github.com/hongbo-miao/hongbomiao.com/issues/824
        'service-worker': ['warn', { minScore: 0 }],
        'third-party-cookies': ['warn', { minScore: 0 }],
        'total-byte-weight': ['warn', { minScore: 0 }],
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
