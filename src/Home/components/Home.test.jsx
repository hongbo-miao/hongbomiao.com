import React from 'react';
import ReactDOM from 'react-dom';

import HmHome from './Home';


describe('Home', () => {
  test('render Home', () => {
    const div = document.createElement('div');
    ReactDOM.render(<HmHome />, div);
    ReactDOM.unmountComponentAtNode(div);
  });
});
