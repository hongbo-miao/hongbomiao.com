import React from 'react';

import HmFooter from './Footer';
import TestUtils from '../../shared/utils/testUtils';

describe('Footer', () => {
  test('render Footer', () => {
    TestUtils.testComponent(<HmFooter />);
  });
});
