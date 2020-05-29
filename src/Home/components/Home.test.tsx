import React from 'react';

import HmHome from './Home';
import TestUtils from '../../shared/utils/testUtils';

describe('Home', () => {
  test('render Home', () => {
    TestUtils.testComponent(<HmHome />);
  });
});
