import { Layout, Text } from '@ui-kitten/components';
import React from 'react';
import { StyleSheet } from 'react-native';
import { connect, ConnectedProps } from 'react-redux';
import RootState from '../../shared/types/RootState.type';
import MeAction from '../actions/MeAction';
import meQuery from '../queries/meQuery';

const connector = connect(
  (state: RootState) => ({
    me: state.me,
  }),
  {
    queryMe: MeAction.queryMe,
  },
);

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

type Props = ConnectedProps<typeof connector>;

function Home(props: Props) {
  const { me, queryMe } = props;

  React.useEffect(() => {
    queryMe(meQuery);
  }, [queryMe]);

  const { bio, name } = me;

  return (
    <Layout style={styles.hmHome}>
      <Text style={styles.hmName} testID="name">
        {name}
      </Text>
      <Text style={styles.hmBio} testID="bio">
        {bio}
      </Text>
    </Layout>
  );
}

export default connector(Home);
