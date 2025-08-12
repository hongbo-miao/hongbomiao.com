import { render } from '@testing-library/react';
import HmCopyright from './Copyright';

describe('Copyright', () => {
  test('render Copyright', () => {
    render(<HmCopyright year={1990} />);
  });
});
