import React, { lazy, useEffect } from 'react';
import { connect, ConnectedProps } from 'react-redux';
import config from '../../config';
import HealthAction from '../../health/actions/HealthAction';
import pingSubscription from '../../health/queries/pingSubscription';
import HmLazyComponent from '../../shared/components/LazyComponent';
import RootState from '../../shared/types/RootState.type';
import analytics from '../../shared/utils/analytics';
import MeAction from '../actions/MeAction';
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import springWaltzMP3 from '../audio/spring-waltz.mp3';
import hatAVIF from '../images/hat.avif';
import hatPNG from '../images/hat.png';
import magicAVIF from '../images/magic.avif';
import magicPNG from '../images/magic.png';
import meQuery from '../queries/meQuery';
import styles from './Home.module.css';

const HmAudioPlayer = lazy(() => import('../../shared/components/AudioPlayer'));
const HmImage = lazy(() => import('../../shared/components/Image'));
const HmSparkles = lazy(() => import('../../shared/components/Sparkles'));

const connector = connect(
  (state: RootState) => ({
    me: state.me,
  }),
  {
    queryMe: MeAction.queryMe,
    subscribePing: HealthAction.subscribePing,
  }
);

type Props = ConnectedProps<typeof connector>;

const Home: React.FC<Props> = (props) => {
  const { me, queryMe, subscribePing } = props;

  useEffect(() => {
    queryMe(meQuery);
    subscribePing(pingSubscription);

    analytics.page();
  }, [queryMe]);

  const { bio, name } = me;

  return (
    <div className={styles.hmHome}>
      <div className={`container is-max-desktop ${styles.hmContainer}`}>
        <div className={styles.hmNameContainer}>
          <h1 className={styles.hmName}>{name}</h1>
          <HmLazyComponent>
            <HmAudioPlayer audioSrc={springWaltzMP3} />
          </HmLazyComponent>
        </div>
        <HmLazyComponent>
          <HmSparkles>
            <a className={styles.hmBioContainer} href={config.githubURL} target="_blank" rel="noopener noreferrer">
              <HmImage
                avifSrc={hatAVIF}
                fallbackSrc={hatPNG}
                style={{ height: '22px', width: '22px' }}
                alt="Magical Hat"
              />
              <div className={styles.hmBio}>{bio}</div>
              <HmImage
                avifSrc={magicAVIF}
                fallbackSrc={magicPNG}
                style={{ height: '22px', width: '22px' }}
                alt="Magic"
              />
            </a>
          </HmSparkles>
        </HmLazyComponent>
      </div>
    </div>
  );
};

export default connector(Home);
