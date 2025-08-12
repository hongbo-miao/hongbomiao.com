import { render } from '@testing-library/react';
import HmLazyComponent from './LazyComponent';

describe('Suspense', () => {
  test('render Suspense', () => {
    render(<HmLazyComponent>Hello, World!</HmLazyComponent>);
  });
});
