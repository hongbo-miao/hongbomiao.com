import React from 'react';
import ReactDOM from 'react-dom';

import HmLoading from './Loading';


describe('Loading', () => {
  test('render Loading', () => {
    const div = document.createElement('div');
    ReactDOM.render(<HmLoading />, div);
    ReactDOM.unmountComponentAtNode(div);
  });
});
