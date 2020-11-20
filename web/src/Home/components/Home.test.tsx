import { render } from '@testing-library/react';
import React from 'react';
import { Provider } from 'react-redux';
import mockedStore from '../../shared/utils/mockedStore';
import HmHome from './Home';

describe('Home', () => {
  test('render Home', () => {
    render(
      <Provider store={mockedStore}>
        <HmHome />
      </Provider>
    );
  });
});
