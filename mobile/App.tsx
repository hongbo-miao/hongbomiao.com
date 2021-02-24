import * as eva from '@eva-design/eva';
import { ApplicationProvider } from '@ui-kitten/components';
import AppLoading from 'expo-app-loading';
import { useFonts } from 'expo-font';
import React from 'react';
import HmHome from './Home/components/Home';

const App: React.FC = () => {
  const [isFontLoaded] = useFonts({
    // eslint-disable-next-line global-require
    NeoSansProRegular: require('./assets/fonts/NeoSansProRegular.otf'),
    // eslint-disable-next-line global-require
    NeoSansProBold: require('./assets/fonts/NeoSansProBold.otf'),
  });

  if (!isFontLoaded) {
    return <AppLoading />;
  }

  return (
    // eslint-disable-next-line react/jsx-props-no-spreading
    <ApplicationProvider {...eva} theme={eva.light}>
      <HmHome />
    </ApplicationProvider>
  );
};

export default App;
