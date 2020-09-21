module.exports = {
  extends: [
    'stylelint-config-standard', // stylelint-config-standard
    'stylelint-prettier/recommended', // stylelint-config-prettier. Turn off all rules that are unnecessary or might conflict with Prettier. Make sure to put it last, so it will override other configs.
  ],
};
