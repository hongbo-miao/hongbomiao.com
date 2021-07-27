import React from 'react';
import { Link, Redirect, Route, Switch } from 'react-router-dom';
import useMe from '../../auth/hooks/useMe';
import HmLazyComponent from '../../shared/components/LazyComponent';
import Paths from '../../shared/utils/paths';
import styles from './Lab.module.css';

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
        <aside className="menu">
          <p className="menu-label">General</p>
          <ul className="menu-list">
            <li>
              <Link to={Paths.welcomePath}>Welcome</Link>
            </li>
          </ul>
          <p className="menu-label">Experiments</p>
          <ul className="menu-list">
            <li>
              <Link to={Paths.opaPath}>OPA</Link>
            </li>
            <li>
              <Link to={Paths.opalPath}>OPAL</Link>
            </li>
          </ul>
        </aside>
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
