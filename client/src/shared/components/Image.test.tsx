import { shallow } from 'enzyme';
import React from 'react';

import TestUtils from '../utils/testUtils';
import HmImage from './Image';

describe('Image', () => {
  const alt = 'Hello';
  const src = 'https://example.com/image.png';
  const webpSrc = 'https://example.com/image.webp';
  const component = <HmImage alt="Hello" src={src} webpSrc={webpSrc} />;

  test('render Image', () => {
    TestUtils.testComponent(component);
  });

  test('picture contains webp', () => {
    const wrapper = shallow(component);
    expect(wrapper.contains(<source type="image/webp" srcSet={webpSrc} />)).toBe(true);
  });

  test('picture contains fallback img', () => {
    const wrapper = shallow(component);
    expect(wrapper.contains(<img src={src} alt={alt} />)).toBe(true);
  });
});
