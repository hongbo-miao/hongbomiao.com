import { render } from '@testing-library/react';
import React from 'react';
import HmImage from './Image';

describe('Image', () => {
  const webpSrc = 'https://example.com/image.webp';
  const fallbackSrc = 'https://example.com/image.png';
  const component = (
    <HmImage webpSrc={webpSrc} fallbackSrc={fallbackSrc} style={{ height: '1px', width: '1px' }} alt="Hello, World!" />
  );

  test('render Image', () => {
    render(component);
  });

  test('picture contains webp', () => {
    const { container } = render(component);
    expect(container.getElementsByTagName('source')).toHaveLength(1);
  });

  test('picture contains fallback img', () => {
    const { container } = render(component);
    expect(container.getElementsByTagName('img')).toHaveLength(1);
  });
});
