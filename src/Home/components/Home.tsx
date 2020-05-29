import React from 'react';

import Config from '../../config';
import hatPNG from '../images/hat.png';
import hatWebP from '../images/hat.webp';
import magicPNG from '../images/magic.png';
import magicWebP from '../images/magic.webp';
import styles from './Home.module.css';
import LazyComponent from '../../shared/components/LazyComponent';
import HmImage from '../../shared/components/Image';

const Home: React.FC = () => (
  <div className={styles.hmHome}>
    <div className={`container ${styles.hmContainer}`}>
      <h1 className={styles.hmTitle}>HONGBO MIAO</h1>
      <a className={styles.hmContent} href={Config.githubUrl} target="_blank" rel="noopener noreferrer">
        <LazyComponent>
          <HmImage className={styles.hmEmoji} alt="Magical Hat" src={hatPNG} webpSrc={hatWebP} />
        </LazyComponent>
        <div className={styles.hmText}>Making magic happen</div>
        <LazyComponent>
          <HmImage className={styles.hmEmoji} alt="Magic" src={magicPNG} webpSrc={magicWebP} />
        </LazyComponent>
      </a>
    </div>
  </div>
);

export default Home;
