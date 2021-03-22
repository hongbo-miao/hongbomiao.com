const fs = require('fs');
const path = require('path');
const solc = require('solc');

const inboxPath = path.resolve(__dirname, 'contracts', 'Storage.sol');
const source = fs.readFileSync(inboxPath, 'utf-8');

const input = {
  language: 'Solidity',
  sources: {
    'Storage.sol': {
      content: source,
    },
  },
  settings: {
    outputSelection: {
      '*': {
        '*': ['*'],
      },
    },
  },
};

module.exports = JSON.parse(solc.compile(JSON.stringify(input)));
