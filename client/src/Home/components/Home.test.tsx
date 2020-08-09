import { render } from '@testing-library/react';
import React from 'react';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import HmHome from './Home';

describe('Home', () => {
  test('render Home', () => {
    const mockStore = configureStore();
    const store = mockStore();

    render(
      <Provider store={store}>
        <HmHome />
      </Provider>
    );
  });
});
