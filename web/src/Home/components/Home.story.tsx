import { storiesOf } from '@storybook/react';
import React from 'react';
import { Provider } from 'react-redux';
import mockedStore from '../../shared/utils/mockedStore';
import HmHome from './Home';

storiesOf('Home', module).add('default', () => (
  <Provider store={mockedStore}>
    <HmHome />
  </Provider>
));
