import { render } from '@testing-library/react';
import HmLoading from '@/shared/components/Loading';

describe('Loading', () => {
  test('render Loading', () => {
    render(<HmLoading />);
  });
});
