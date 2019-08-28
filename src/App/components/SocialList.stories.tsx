import React from 'react';
import { storiesOf } from '@storybook/react';

import WEBSITES from '../fixtures/websites';
import HmSocialList from './SocialList';

storiesOf('SocialList', module)
  .add('default', () => <HmSocialList websites={WEBSITES} />)
  .add('empty', () => <HmSocialList websites={[]} />);
