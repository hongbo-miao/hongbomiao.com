import React from 'react';
import ReactDOM from 'react-dom';

import HmApp from './App';


describe('App', () => {
  test('render App', () => {
    const div = document.createElement('div');
    ReactDOM.render(<HmApp />, div);
    ReactDOM.unmountComponentAtNode(div);
  });
});
