import React from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import useMe from '../../auth/hooks/useMe';
import HmLazyComponent from '../../shared/components/LazyComponent';
import Paths from '../../shared/utils/paths';
import styles from './Lab.module.css';
import HmMenu from './Menu';
import HmNavbar from './Navbar';

const HmOPAExperiment = React.lazy(() => import('./OPAExperiment'));
const HmOPALExperiment = React.lazy(() => import('./OPALExperiment'));
const HmWelcome = React.lazy(() => import('./Welcome'));

function Lab() {
  const { me } = useMe();

  if (me == null) {
    return <Navigate to="/signin" replace />;
  }

  return (
    <div className="container is-max-desktop">
      <HmNavbar />
      <div className={styles.hmBody}>
        <HmMenu />
        <HmLazyComponent>
          <Routes>
            <Route index element={<HmWelcome />} />
            <Route path={Paths.Lab.opaPath} element={<HmOPAExperiment />} />
            <Route path={Paths.Lab.opalPath} element={<HmOPALExperiment />} />
          </Routes>
        </HmLazyComponent>
      </div>
    </div>
  );
}

export default Lab;
