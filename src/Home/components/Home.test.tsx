import React from 'react';

import TestUtils from '../../shared/utils/testUtils';
import HmHome from './Home';

describe('Home', () => {
  test('render Home', () => {
    TestUtils.testComponent(<HmHome />);
  });
});
