import React from 'react';

import hatImage from '../images/hat.png';
import magicImage from '../images/magic.png';
import styles from './Home.module.css';

const Home: React.FC = () => (
  <div className={styles.hmHome}>
    <div className={`container ${styles.hmContainer}`}>
      <h1 className="title">HONGBO MIAO</h1>
      <div className={styles.hmContent}>
        <img className={styles.hmEmoji} src={hatImage} alt="Magical Hat" />
        <div className={styles.hmText}>Making magic happen</div>
        <img className={styles.hmEmoji} src={magicImage} alt="Magic" />
      </div>
    </div>
  </div>
);

export default Home;
