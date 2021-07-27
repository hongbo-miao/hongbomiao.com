import { storiesOf } from '@storybook/react';
import React from 'react';
import WEBSITES from '../../Home/fixtures/WEBSITES';
import HmSocialList from './SocialList';

storiesOf('SocialList', module)
  .add('default', () => <HmSocialList websites={WEBSITES} />)
  .add('empty', () => <HmSocialList websites={[]} />);
