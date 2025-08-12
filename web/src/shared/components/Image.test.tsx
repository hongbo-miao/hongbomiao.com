import { render } from '@testing-library/react';
import HmImage from './Image';

describe('Image', () => {
  const avifSrc = 'https://example.com/image.avif';
  const fallbackSrc = 'https://example.com/image.png';
  const component = (
    <HmImage avifSrc={avifSrc} fallbackSrc={fallbackSrc} style={{ height: '1px', width: '1px' }} alt="Hello, World!" />
  );

  test('render Image', () => {
    render(component);
  });

  test('picture contains avif', () => {
    const { container } = render(component);
    // eslint-disable-next-line testing-library/no-container,testing-library/no-node-access
    expect(container.getElementsByTagName('source')).toHaveLength(1);
  });

  test('picture contains fallback img', () => {
    const { container } = render(component);
    // eslint-disable-next-line testing-library/no-container,testing-library/no-node-access
    expect(container.getElementsByTagName('img')).toHaveLength(1);
  });
});
