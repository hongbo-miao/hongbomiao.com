import React from 'react';
import { storiesOf } from '@storybook/react';

import HmCopyright from './Copyright';


storiesOf('Copyright', module)
  .add('default', () => <HmCopyright year={1990} />);
