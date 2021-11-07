import React from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import useMe from '../../auth/hooks/useMe';
import HmLazyComponent from '../../shared/components/LazyComponent';
import Paths from '../../shared/utils/paths';
import styles from './Lab.module.css';

const HmMenu = React.lazy(() => import('./Menu'));
const HmNavbar = React.lazy(() => import('./Navbar'));
const HmOPAExperiment = React.lazy(() => import('./OPAExperiment'));
const HmOPALExperiment = React.lazy(() => import('./OPALExperiment'));
const HmWelcome = React.lazy(() => import('./Welcome'));

const Lab: React.VFC = () => {
  const { me } = useMe();

  if (me == null) {
    return <Navigate to="/signin" />;
  }

  return (
    <div className="container is-max-desktop">
      <HmLazyComponent>
        <HmNavbar />
      </HmLazyComponent>
      <div className={styles.hmBody}>
        <HmLazyComponent>
          <HmMenu />
        </HmLazyComponent>
        <HmLazyComponent>
          <Routes>
            <Route path={Paths.welcomePath} element={<HmWelcome />} />
            <Route path={Paths.opaPath} element={<HmOPAExperiment />} />
            <Route path={Paths.opalPath} element={<HmOPALExperiment />} />
          </Routes>
        </HmLazyComponent>
      </div>
    </div>
  );
};

export default Lab;
