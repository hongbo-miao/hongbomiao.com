// https://stylelint.io/user-guide/configure

/**
 * @type {import("stylelint").Config}
 */
export default {
  extends: [
    'stylelint-config-standard', // stylelint-config-standard
    'stylelint-config-recess-order', // stylelint-config-recess-order
    'stylelint-prettier/recommended', // stylelint-config-prettier. Turn off all rules that are unnecessary or might conflict with Prettier. Make sure to put it last, so it will override other configs.
  ],
  plugins: ['stylelint-high-performance-animation'], // stylelint-high-performance-animation
  rules: {
    // stylelint-config-standard
    'keyframes-name-pattern': null,
    'selector-class-pattern': null,
    'import-notation': null,

    // stylelint-high-performance-animation
    'plugin/no-low-performance-animation-properties': true,
  },
};
