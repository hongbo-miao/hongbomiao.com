import React from 'react';

import HmLazyComponent from './LazyComponent';
import TestUtils from '../utils/testUtils';

describe('Suspense', () => {
  test('render Suspense', () => {
    TestUtils.testComponent(<HmLazyComponent>hello</HmLazyComponent>);
  });
});
