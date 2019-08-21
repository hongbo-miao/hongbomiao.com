module.exports = {
  parser: 'babel-eslint',
  env: {
    browser: true,
    jest: true
  },
  extends: [
    'airbnb'
  ],
  rules: {
    'import/no-extraneous-dependencies': ['error', {
      devDependencies: true,
    }],
  }
};
