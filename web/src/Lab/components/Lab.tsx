import React from 'react';
import { Redirect } from 'react-router-dom';
import useMe from '../../auth/hooks/useMe';
import styles from './Lab.module.css';

const HmNavbar = React.lazy(() => import('./Navbar'));

const Lab: React.VFC = () => {
  const { me } = useMe();

  if (me == null) {
    return <Redirect to="/signin" />;
  }

  return (
    <>
      <HmNavbar />
      <div className={styles.hmLab}>
        <div className={`container is-max-desktop ${styles.hmContainer}`}>Lab</div>
      </div>
    </>
  );
};

export default Lab;
