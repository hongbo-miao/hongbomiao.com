import { render } from '@testing-library/react';
import React from 'react';
import HmLoading from './Loading';

describe('Loading', () => {
  test('render Loading', () => {
    render(<HmLoading />);
  });
});
