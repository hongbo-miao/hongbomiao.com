import * as eva from '@eva-design/eva';
import { ApplicationProvider } from '@ui-kitten/components';
import React from 'react';
import { Provider } from 'react-redux';
import renderer from 'react-test-renderer';
import mockedStore from '../../shared/utils/mockedStore';
import HmHome from './Home';

describe('<HmHome />', () => {
  test('HmHome has 1 child', () => {
    const tree = renderer
      .create(
        <Provider store={mockedStore}>
          {/* eslint-disable-next-line react/jsx-props-no-spreading */}
          <ApplicationProvider {...eva} theme={eva.light}>
            <HmHome />
          </ApplicationProvider>
        </Provider>
      )
      .toJSON();

    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    expect(tree.children.length).toBe(1);
  });
});
