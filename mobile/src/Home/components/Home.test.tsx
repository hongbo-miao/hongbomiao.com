import * as eva from '@eva-design/eva';
import { render } from '@testing-library/react-native';
import { ApplicationProvider } from '@ui-kitten/components';
import React from 'react';
import { Provider } from 'react-redux';
import mockedStore from '../../shared/utils/mockedStore';
import HmHome from './Home';

describe('<HmHome />', () => {
  test('Home', () => {
    const { getByTestId, toJSON } = render(
      <Provider store={mockedStore}>
        {/* eslint-disable-next-line react/jsx-props-no-spreading */}
        <ApplicationProvider {...eva} theme={eva.light}>
          <HmHome />
        </ApplicationProvider>
      </Provider>
    );
    expect(getByTestId('bio').props.children).toBe('Making magic happen');
    expect(toJSON()).toMatchSnapshot();
  });
});
