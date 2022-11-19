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
    fontFamily: 'NeoSansProBold',
    fontSize: 35,
    marginTop: 20,
    textTransform: 'uppercase',
  },
  hmBio: {
    fontFamily: 'NeoSansProRegular',
    fontSize: 20,
    marginTop: 40,
  },
});

function Home() {
  return (
    <Layout style={styles.hmHome}>
      <Text style={styles.hmName} testID="name">
        Hongbo Miao
      </Text>
      <Text style={styles.hmBio} testID="bio">
        Making magic happen
      </Text>
    </Layout>
  );
}

export default Home;
