import { render } from '@testing-library/react';
import React from 'react';
import HmApp from './App';

describe('App', () => {
  test('render App', () => {
    render(<HmApp />);
  });
});
