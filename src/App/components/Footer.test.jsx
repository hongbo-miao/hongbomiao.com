import React from 'react';
import { shallow } from 'enzyme';

import TestUtils from '../../shared/utils/testUtils';
import HmFooter from './Footer';


describe('Footer', () => {
  test('render Footer', () => {
    TestUtils.testComponent(<HmFooter />);
  });

  test('render .hm-copyright', () => {
    const wrapper = shallow(<HmFooter />);
    expect(wrapper.find('.hm-copyright')).toHaveLength(1);
  });
});
