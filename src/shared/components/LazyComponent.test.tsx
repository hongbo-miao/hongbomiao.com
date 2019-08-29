import React from 'react';

import TestUtils from '../utils/testUtils';
import LazyComponent from './LazyComponent';

describe('Suspense', () => {
  test('render Suspense', () => {
    TestUtils.testComponent(<LazyComponent>hello</LazyComponent>);
  });
});
