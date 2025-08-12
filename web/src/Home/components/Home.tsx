import React from 'react';
import config from '../../config';
import useHealthSubscription from '../../health/hooks/useHealthSubscription';
import HmAudioPlayer from '../../shared/components/AudioPlayer';
import HmImage from '../../shared/components/Image';
import HmSparkles from '../../shared/components/Sparkles';
import analytics from '../../shared/utils/analytics';
import springWaltzMP3 from '../audio/spring-waltz.mp3';
import hatAVIF from '../images/hat.avif';
import hatPNG from '../images/hat.png';
import magicAVIF from '../images/magic.avif';
import magicPNG from '../images/magic.png';
import HmFooter from './Footer';
import styles from './Home.module.css';

function Home() {
  useHealthSubscription();

  React.useEffect(() => {
    analytics.page();
  }, []);

  // Static content instead of auth-based data
  const name = 'Hongbo Miao';
  const bio = 'Making magic happen';

  return (
    <div className={styles.hmHome}>
      <div className={styles.hmBody}>
        <div className={`container is-max-desktop ${styles.hmContainer}`}>
          <div className={styles.hmNameContainer}>
            <h1 className={styles.hmName}>{name}</h1>
            <div className={styles.hmAudioPlayerWrapper}>
              <HmAudioPlayer audioSrc={springWaltzMP3} />
            </div>
          </div>
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
        </div>
      </div>
      <HmFooter />
    </div>
  );
}

export default Home;
