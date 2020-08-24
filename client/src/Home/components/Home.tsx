import React, { useEffect } from 'react';
import { connect, ConnectedProps } from 'react-redux';
import Config from '../../Config';
import HmImage from '../../shared/components/Image';
import HmLazyComponent from '../../shared/components/LazyComponent';
import HmSparkles from '../../shared/components/Sparkles';
import RootState from '../../shared/types/RootState.type';
import MeAction from '../actions/Me.action';
import hatPNG from '../images/hat.png';
import hatWebP from '../images/hat.webp';
import magicPNG from '../images/magic.png';
import magicWebP from '../images/magic.webp';
import meQuery from '../queries/me.query';
import styles from './Home.module.css';

const connector = connect(
  (state: RootState) => ({
    me: state.me,
  }),
  {
    fetchMe: MeAction.fetchMe,
  }
);

type Props = ConnectedProps<typeof connector>;

const Home: React.FC<Props> = (props) => {
  const { me, fetchMe } = props;

  useEffect(() => {
    fetchMe(meQuery);
  }, [fetchMe]);

  const { name, slogan } = me;

  return (
    <div className={styles.hmHome}>
      <div className={`container ${styles.hmContainer}`}>
        <h1 className={styles.hmTitle}>{name}</h1>
        <HmSparkles>
          <a className={styles.hmContent} href={Config.githubURL} target="_blank" rel="noopener noreferrer">
            <HmLazyComponent>
              <HmImage
                webpSrc={hatWebP}
                fallbackSrc={hatPNG}
                style={{ height: '22px', width: '22px' }}
                alt="Magical Hat"
              />
            </HmLazyComponent>
            <div className={styles.hmText}>{slogan}</div>
            <HmLazyComponent>
              <HmImage
                webpSrc={magicWebP}
                fallbackSrc={magicPNG}
                style={{ height: '22px', width: '22px' }}
                alt="Magic"
              />
            </HmLazyComponent>
          </a>
        </HmSparkles>
      </div>
    </div>
  );
};

export default connector(Home);
