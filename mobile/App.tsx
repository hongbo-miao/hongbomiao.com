import * as eva from '@eva-design/eva';
import { ApplicationProvider } from '@ui-kitten/components';
import * as Font from 'expo-font';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { StyleSheet, View } from 'react-native';
import HmHome from './src/Home/components/Home';

// Keep the splash screen visible while we fetch resources
SplashScreen.preventAutoHideAsync();

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});

function App() {
  const [isAppReady, setIsAppReady] = React.useState(false);

  React.useEffect(() => {
    const prepare = async () => {
      try {
        await Font.loadAsync({
          // eslint-disable-next-line global-require
          NeoSansProRegular: require('./assets/fonts/NeoSansProRegular.otf'),
          // eslint-disable-next-line global-require
          NeoSansProBold: require('./assets/fonts/NeoSansProBold.otf'),
        });
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn(err);
      } finally {
        setIsAppReady(true);
      }
    };

    prepare();
  }, []);

  const onLayoutRootView = React.useCallback(async () => {
    if (isAppReady) {
      /*
       * This tells the splash screen to hide immediately! If we call this after
       * `setIsAppReady`, then we may see a blank screen while the app is
       * loading its initial state and rendering its first pixels. So instead,
       * we hide the splash screen once we know the root view has already
       * performed layout.
       */
      await SplashScreen.hideAsync();
    }
  }, [isAppReady]);

  if (!isAppReady) {
    return null;
  }
  return (
    // eslint-disable-next-line react/jsx-props-no-spreading
    <ApplicationProvider {...eva} theme={eva.light}>
      <View style={styles.container} onLayout={onLayoutRootView}>
        <StatusBar />
        <HmHome />
      </View>
    </ApplicationProvider>
  );
}
export default App;
