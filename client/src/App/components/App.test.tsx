import React from 'react';
import TestUtils from '../../shared/utils/testUtils';
import HmApp from './App';

describe('App', () => {
  test('render App', () => {
    TestUtils.testComponent(<HmApp />);
  });
});
