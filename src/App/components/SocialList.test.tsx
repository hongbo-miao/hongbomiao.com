import React from 'react';
import { shallow } from 'enzyme';

import HmSocialList from './SocialList';
import TestUtils from '../../shared/utils/testUtils';
import WEBSITES from '../fixtures/websites';

describe('SocialList', () => {
  test('render SocialList', () => {
    TestUtils.testComponent(<HmSocialList websites={WEBSITES} />);
  });

  test('render .level-item', () => {
    const wrapper = shallow(<HmSocialList websites={WEBSITES} />);
    expect(wrapper.find('.level-item')).toHaveLength(WEBSITES.length);
  });
});
