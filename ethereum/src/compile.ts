import fs from 'fs';
import path from 'path';
import solc from 'solc';

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

export default JSON.parse(solc.compile(JSON.stringify(input)));
