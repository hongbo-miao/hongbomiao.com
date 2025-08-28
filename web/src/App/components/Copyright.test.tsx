import { render } from '@testing-library/react';
import HmCopyright from '@/App/components/Copyright';

describe('Copyright', () => {
  test('render Copyright', () => {
    render(<HmCopyright year={1990} />);
  });
});
