module.exports = {
  addons: ['@storybook/addon-actions', '@storybook/addon-links', '@storybook/preset-create-react-app'],
  stories: ['../src/**/*.story.tsx'],
  core: {
    builder: 'webpack5'
  }
};