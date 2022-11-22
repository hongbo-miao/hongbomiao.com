import * as eva from '@eva-design/eva';
import { render } from '@testing-library/react-native';
import { ApplicationProvider } from '@ui-kitten/components';
import React from 'react';
import HmHome from './Home';

describe('<HmHome />', () => {
  test('Home', () => {
    const { getByTestId, toJSON } = render(
      // eslint-disable-next-line react/jsx-props-no-spreading
      <ApplicationProvider {...eva} theme={eva.light}>
        <HmHome />
      </ApplicationProvider>,
    );
    expect(getByTestId('bio').props.children).toBe('Making magic happen');
    expect(toJSON()).toMatchSnapshot();
  });
});
