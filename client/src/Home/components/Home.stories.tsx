import { storiesOf } from '@storybook/react';
import React from 'react';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import { RootState } from '../../reducer';
import Me from '../types/Me.type';
import HmHome from './Home';

const mockStore = configureStore();
const me: Me = {
  name: 'Hongbo Miao',
  slogan: 'Making magic happen',
};
const initialState: RootState = { me };
const store = mockStore(initialState);

storiesOf('Home', module).add('default', () => (
  <Provider store={store}>
    <HmHome />
  </Provider>
));
