import { shallow } from 'enzyme';
import React from 'react';
import TestUtils from '../../shared/utils/testUtils';
import WEBSITES from '../fixtures/websites';
import HmSocialList from './SocialList';

describe('SocialList', () => {
  test('render SocialList', () => {
    TestUtils.testComponent(<HmSocialList websites={WEBSITES} />);
  });

  test('render .level-item', () => {
    const wrapper = shallow(<HmSocialList websites={WEBSITES} />);
    expect(wrapper.find('.level-item')).toHaveLength(WEBSITES.length);
  });
});
