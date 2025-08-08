import { Outlet, useRouter } from '@tanstack/react-router';
import React from 'react';
import useMe from '../../auth/hooks/useMe';
import styles from './Lab.module.css';
import HmMenu from './Menu';
import HmNavbar from './Navbar';

function Lab() {
  const { me } = useMe();
  const router = useRouter();

  if (me == null) {
    router.history.push('/signin');
    return null;
  }

  return (
    <div className="container is-max-desktop">
      <HmNavbar />
      <div className={styles.hmBody}>
        <HmMenu />
        <Outlet />
      </div>
    </div>
  );
}

export default Lab;
