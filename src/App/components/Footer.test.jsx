import React from 'react';
import ReactDOM from 'react-dom';

import HmFooter from './Footer';


test('render Footer', () => {
  const div = document.createElement('div');
  ReactDOM.render(<HmFooter />, div);
  ReactDOM.unmountComponentAtNode(div);
});
