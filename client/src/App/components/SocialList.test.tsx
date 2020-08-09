import { render } from '@testing-library/react';
import React from 'react';
import WEBSITES from '../fixtures/websites';
import HmSocialList from './SocialList';

describe('SocialList', () => {
  const component = <HmSocialList websites={WEBSITES} />;

  test('render SocialList', () => {
    render(component);
  });

  test('render .level-item', () => {
    const { container } = render(component);
    expect(container.getElementsByTagName('img')).toHaveLength(WEBSITES.length);
  });
});
