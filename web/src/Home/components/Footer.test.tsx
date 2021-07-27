import { render } from '@testing-library/react';
import React from 'react';
import HmFooter from './Footer';

describe('Footer', () => {
  test('render Footer', () => {
    render(<HmFooter />);
  });
});
