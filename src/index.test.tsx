import React from 'react';
import ReactDOM from 'react-dom';

import HmApp from './App/components/App';


jest.mock('react-dom', () => ({ render: jest.fn() }));

describe('index', () => {
  test('render index', () => {
    const div = document.createElement('div');
    div.id = 'root';
    document.body.appendChild(div);

    // eslint-disable-next-line global-require
    require('./index.tsx');
    expect(ReactDOM.render).toHaveBeenCalledWith(<HmApp />, div);
  });
});
