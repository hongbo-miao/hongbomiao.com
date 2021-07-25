import React from 'react';
import styles from './SignIn.module.css';

const SignIn: React.VFC = () => {
  return (
    <div className={styles.hmSignIn}>
      <div className={`container is-max-desktop ${styles.hmContainer}`}>Sign In</div>
    </div>
  );
};

export default SignIn;
