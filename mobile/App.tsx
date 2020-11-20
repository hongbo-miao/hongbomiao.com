import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

const App: React.FC = () => {
  return (
    <View style={styles.container}>
      <Text>Hongbo Miao</Text>
      <Text>Making magic happen</Text>
    </View>
  );
};

export default App;
