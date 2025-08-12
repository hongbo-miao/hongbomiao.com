import { render } from '@testing-library/react';
import WEBSITES from '../../Home/fixtures/WEBSITES';
import HmSocialList from './SocialList';

describe('SocialList', () => {
  const component = <HmSocialList websites={WEBSITES} />;

  test('render SocialList', () => {
    render(component);
  });

  test('render .level-item', () => {
    const { container } = render(component);
    // eslint-disable-next-line testing-library/no-container,testing-library/no-node-access
    expect(container.getElementsByTagName('img')).toHaveLength(WEBSITES.length);
  });
});
