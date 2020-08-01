import React from 'react';

import TestUtils from '../../shared/utils/testUtils';
import HmCopyright from './Copyright';

describe('Copyright', () => {
  test('render Copyright', () => {
    TestUtils.testComponent(<HmCopyright year={1990} />);
  });
});
