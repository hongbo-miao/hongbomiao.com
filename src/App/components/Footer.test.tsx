import React from 'react';

import TestUtils from '../../shared/utils/testUtils';
import HmFooter from './Footer';

describe('Footer', () => {
  test('render Footer', () => {
    TestUtils.testComponent(<HmFooter />);
  });
});
