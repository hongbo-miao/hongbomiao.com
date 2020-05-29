import React from 'react';

import HmApp from './App';
import TestUtils from '../../shared/utils/testUtils';

describe('App', () => {
  test('render App', () => {
    TestUtils.testComponent(<HmApp />);
  });
});
