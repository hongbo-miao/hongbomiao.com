module.exports = {
  plugins: [
    'react',
  ],
  extends: [
    'plugin:react/recommended',
 ],
  env: {
    browser: true,
  },
  rules: {
    'react/jsx-filename-extension': [1, {
      extensions: ['.jsx', '.tsx'],
    }],
  },
};
