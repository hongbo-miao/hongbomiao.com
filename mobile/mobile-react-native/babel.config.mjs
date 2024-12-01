export default (api) => {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
  };
};
