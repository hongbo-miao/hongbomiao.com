import React from 'react';
import { Redirect, Route, Switch } from 'react-router-dom';
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
    return <Redirect to="/signin" />;
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
          <Switch>
            <Route exact path={Paths.welcomePath} component={HmWelcome} />
            <Route exact path={Paths.opaPath} component={HmOPAExperiment} />
            <Route exact path={Paths.opalPath} component={HmOPALExperiment} />
          </Switch>
        </HmLazyComponent>
      </div>
    </div>
  );
};

export default Lab;
