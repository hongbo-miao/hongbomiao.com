import React from 'react';

import Config from '../../config';
import hatImage from '../images/hat.png';
import magicImage from '../images/magic.png';
import styles from './Home.module.css';

const Home: React.FC = () => (
  <div className={styles.hmHome}>
    <div className={`container ${styles.hmContainer}`}>
      <h1 className={styles.hmTitle}>HONGBO MIAO</h1>
      <a className={styles.hmContent} href={Config.githubUrl} target="_blank" rel="noopener noreferrer">
        <img className={styles.hmEmoji} src={hatImage} alt="Magical Hat" />
        <div className={styles.hmText}>Making magic happen</div>
        <img className={styles.hmEmoji} src={magicImage} alt="Magic" />
      </a>
    </div>
  </div>
);

export default Home;
