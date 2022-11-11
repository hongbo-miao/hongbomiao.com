import React from 'react';
import styles from './Welcome.module.css';

function Welcome() {
  return (
    <div className={styles.hmWelcome}>
      <div className={`container is-max-desktop ${styles.hmContainer}`}>
        <div>Welcome to the Lab!</div>
      </div>
    </div>
  );
}

export default Welcome;
