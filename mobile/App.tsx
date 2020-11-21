// eslint-disable-next-line camelcase
import { useFonts, OpenSans_400Regular } from '@expo-google-fonts/open-sans';
import AppLoading from 'expo/build/launch/AppLoadingNativeWrapper';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    fontFamily: 'OpenSans_400Regular',
  },
});

const App: React.FC = () => {
  const [fontsLoaded] = useFonts({
    OpenSans_400Regular,
  });

  if (!fontsLoaded) {
    return <AppLoading />;
  }

  return (
    <View style={styles.container}>
      <Text style={styles.text}>Hongbo Miao</Text>
      <Text style={styles.text}>Making magic happen</Text>
    </View>
  );
};

export default App;
