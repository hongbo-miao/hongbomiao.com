import React from 'react';
import { shallow } from 'enzyme';

import TestUtils from '../../shared/utils/testUtils';
import Websites from '../fixtures/websites';
import HmSocialList from './SocialList';


describe('SocialList', () => {
  test('render SocialList', () => {
    TestUtils.testComponent(<HmSocialList websites={Websites} />);
  });

  test('render .hm-social-item', () => {
    const wrapper = shallow(<HmSocialList websites={Websites} />);
    expect(wrapper.find('.hm-social-item')).toHaveLength(Websites.length);
  });
});
