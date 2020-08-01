import { storiesOf } from '@storybook/react';
import React from 'react';

import HmCopyright from './Copyright';

storiesOf('Copyright', module).add('default', () => <HmCopyright year={1990} />);
