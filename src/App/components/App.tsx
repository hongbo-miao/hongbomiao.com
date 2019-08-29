import React, { lazy } from 'react';
import { BrowserRouter, Redirect, Route, Switch } from 'react-router-dom';

import Paths from '../../shared/utils/paths';
import LazyComponent from '../../shared/components/LazyComponent';
import styles from './App.module.css';

const HmFooter = lazy(() => import('./Footer'));
const HmHome = lazy(() => import('../../Home/components/Home'));

const App: React.FC = () => (
  <div className={styles.hmApp}>
    <BrowserRouter>
      <LazyComponent>
        <Switch>
          <Route exact path={Paths.appRootPath} component={HmHome} />
          <Redirect to={Paths.appRootPath} />
        </Switch>
      </LazyComponent>
    </BrowserRouter>

    <LazyComponent>
      <HmFooter />
    </LazyComponent>
  </div>
);

export default App;
