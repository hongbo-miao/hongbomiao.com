// https://prettier.io/docs/en/configuration.html

module.exports = {
  semi: true,
  trailingComma: 'all',
  singleQuote: true,
  printWidth: 120,
  tabWidth: 2,

  overrides: [
    // Solidity
    // https://github.com/prettier-solidity/prettier-plugin-solidity#configuration-file
    {
      files: '*.sol',
      options: {
        printWidth: 80,
        tabWidth: 4,
        useTabs: false,
        singleQuote: false,
        bracketSpacing: false,
      },
    },
  ],

  // XML
  // https://github.com/prettier/plugin-xml#configuration
  xmlWhitespaceSensitivity: 'ignore',
};
