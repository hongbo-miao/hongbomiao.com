import React from 'react';
import ReactDOM from 'react-dom';
import { shallow } from 'enzyme';

import HmSocialList from './SocialList';
import Websites from '../fixtures/websites';


describe('SocialList', () => {
  test('render SocialList', () => {
    const div = document.createElement('div');
    ReactDOM.render(<HmSocialList websites={Websites} />, div);
    ReactDOM.unmountComponentAtNode(div);
  });

  test('render .hm-social-item', () => {
    const wrapper = shallow(<HmSocialList websites={Websites} />);
    expect(wrapper.find('.hm-social-item')).toHaveLength(Websites.length);
  });
});
