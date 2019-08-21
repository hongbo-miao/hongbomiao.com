import React from 'react';
import ReactDOM from 'react-dom';
import { shallow } from 'enzyme';

import HmFooter from './Footer';


describe('Footer', () => {
  test('render Footer', () => {
    const div = document.createElement('div');
    ReactDOM.render(<HmFooter />, div);
    ReactDOM.unmountComponentAtNode(div);
  });

  test('render .hm-copyright', () => {
    const wrapper = shallow(<HmFooter />);
    expect(wrapper.find('.hm-copyright')).toHaveLength(1);
  });
});
