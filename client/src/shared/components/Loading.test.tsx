import React from 'react';

import TestUtils from '../utils/testUtils';
import HmLoading from './Loading';

describe('Loading', () => {
  test('render Loading', () => {
    TestUtils.testComponent(<HmLoading />);
  });
});
