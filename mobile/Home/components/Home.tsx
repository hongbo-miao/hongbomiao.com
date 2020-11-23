import { Layout, Text } from '@ui-kitten/components';
import React from 'react';
import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  hmHome: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  hmName: {
    fontFamily: 'OpenSans_700Bold',
    fontSize: 35,
    marginTop: 20,
    textTransform: 'uppercase',
  },
  hmBio: {
    fontFamily: 'OpenSans_400Regular',
    fontSize: 20,
    marginTop: 20,
  },
});

const Home: React.FC = () => (
  <Layout style={styles.hmHome}>
    <Text style={styles.hmName}>Hongbo Miao</Text>
    <Text style={styles.hmBio}>Making magic happen</Text>
  </Layout>
);

export default Home;
