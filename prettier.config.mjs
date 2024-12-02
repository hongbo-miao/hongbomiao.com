// https://prettier.io/docs/en/configuration.html

/**
 * @type {import("prettier").Config}
 */
export default {
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
