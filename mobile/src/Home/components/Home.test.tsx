import * as eva from '@eva-design/eva';
import { ApplicationProvider } from '@ui-kitten/components';
import React from 'react';
import renderer from 'react-test-renderer';
import HmHome from './Home';

describe('<HmHome />', () => {
  test('HmHome has 1 child', () => {
    const tree = renderer
      .create(
        // eslint-disable-next-line react/jsx-props-no-spreading
        <ApplicationProvider {...eva} theme={eva.light}>
          <HmHome />
        </ApplicationProvider>
      )
      .toJSON();

    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    expect(tree.children.length).toBe(1);
  });
});
