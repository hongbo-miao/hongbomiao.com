module.exports = {
  extends: [
    'stylelint-config-standard', // stylelint-config-standard
    'stylelint-config-recess-order', // stylelint-config-recess-order
    'stylelint-a11y/recommended', // stylelint-a11y
    'stylelint-prettier/recommended', // stylelint-config-prettier. Turn off all rules that are unnecessary or might conflict with Prettier. Make sure to put it last, so it will override other configs.
  ],
  plugins: ['stylelint-high-performance-animation'], // stylelint-high-performance-animation
  rules: {
    'plugin/no-low-performance-animation-properties': true, // stylelint-high-performance-animation
  },
};
