import React from 'react';

import HmCopyright from './Copyright';
import TestUtils from '../../shared/utils/testUtils';

describe('Copyright', () => {
  test('render Copyright', () => {
    TestUtils.testComponent(<HmCopyright year={1990} />);
  });
});
