// https://prettier.io/docs/en/configuration.html

module.exports = {
  semi: true,
  trailingComma: 'all',
  singleQuote: true,
  printWidth: 120,
  tabWidth: 2,
  overrides: [
    {
      files: '*.sol',
      options: {
        printWidth: 120,
        tabWidth: 4,
        useTabs: false,
        singleQuote: false,
        bracketSpacing: false,
        explicitTypes: 'always',
      },
    },
  ],

  // https://github.com/prettier/plugin-xml#configuration
  xmlWhitespaceSensitivity: 'ignore',
};
