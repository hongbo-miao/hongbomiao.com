import { render } from '@testing-library/react';
import React from 'react';
import HmImage from './Image';

describe('Image', () => {
  const src = 'https://example.com/image.png';
  const webpSrc = 'https://example.com/image.webp';
  const component = <HmImage className="hmHello" alt="Hello, World!" src={src} webpSrc={webpSrc} />;

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
