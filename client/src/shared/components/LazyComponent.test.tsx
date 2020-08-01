import React from 'react';

import TestUtils from '../utils/testUtils';
import HmLazyComponent from './LazyComponent';

describe('Suspense', () => {
  test('render Suspense', () => {
    TestUtils.testComponent(<HmLazyComponent>hello</HmLazyComponent>);
  });
});
