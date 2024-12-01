/*
 * ⚠️⚠️⚠️ THIS FILE WAS SCAFFOLDED BY `@grafana/create-plugin`. DO NOT EDIT THIS FILE DIRECTLY. ⚠️⚠️⚠️
 *
 * In order to extend the configuration follow the steps in .config/README.md
 */

/*
 * This utility function is useful in combination with jest `transformIgnorePatterns` config
 * to transform specific packages (e.g.ES modules) in a projects node_modules folder.
 */
const nodeModulesToTransform = (moduleNames) => `node_modules\/(?!(${moduleNames.join('|')})\/)`;

// Array of known nested grafana package dependencies that only bundle an ESM version
const grafanaESModules = ['ol', 'react-colorful'];

module.exports = {
  nodeModulesToTransform,
  grafanaESModules
}
