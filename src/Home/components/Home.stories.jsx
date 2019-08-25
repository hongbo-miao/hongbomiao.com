import React from 'react';
import { storiesOf } from '@storybook/react';

import HmHome from './Home';


storiesOf('Home', module)
  .add('default', () => <HmHome />);
