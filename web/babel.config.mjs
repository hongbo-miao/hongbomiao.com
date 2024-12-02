export default {
  presets: [
    [
      '@babel/preset-env',
      {
        targets: {
          node: 'current',
        },
        modules: false,
      },
    ],
    '@babel/preset-typescript',
    '@babel/react',
  ],
  plugins: ['@babel/plugin-transform-react-display-name'],
};
