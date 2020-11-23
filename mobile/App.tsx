import * as eva from '@eva-design/eva';
// eslint-disable-next-line camelcase
import { useFonts, OpenSans_400Regular, OpenSans_700Bold } from '@expo-google-fonts/open-sans';
import { ApplicationProvider } from '@ui-kitten/components';
import { AppLoading } from 'expo';
import React from 'react';
import HmHome from './Home/components/Home';

const App: React.FC = () => {
  const [isFontLoaded] = useFonts({
    OpenSans_400Regular,
    OpenSans_700Bold,
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
