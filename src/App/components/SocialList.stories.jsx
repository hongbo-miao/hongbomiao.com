import React from 'react';
import { storiesOf } from '@storybook/react';

import Websites from '../fixtures/websites';
import HmSocialList from './SocialList';


storiesOf('SocialList', module)
  .add('default', () => <HmSocialList websites={Websites} />)
  .add('empty', () => <HmSocialList websites={[]} />);
