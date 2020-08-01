import React from 'react';
import { storiesOf } from '@storybook/react';

import HmSocialList from './SocialList';
import WEBSITES from '../fixtures/websites';

storiesOf('SocialList', module)
  .add('default', () => <HmSocialList websites={WEBSITES} />)
  .add('empty', () => <HmSocialList websites={[]} />);
