import React from 'react';

import HmLoading from './Loading';
import TestUtils from '../utils/testUtils';

describe('Loading', () => {
  test('render Loading', () => {
    TestUtils.testComponent(<HmLoading />);
  });
});
