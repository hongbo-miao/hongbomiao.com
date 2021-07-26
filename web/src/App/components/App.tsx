import React from 'react';
import { BrowserRouter, Redirect, Route, Switch } from 'react-router-dom';
import HmLazyComponent from '../../shared/components/LazyComponent';
import Paths from '../../shared/utils/paths';
import styles from './App.module.css';

const HmHome = React.lazy(() => import('../../Home/components/Home'));
const HmSignIn = React.lazy(() => import('../../SignIn/components/SignIn'));
const HmLab = React.lazy(() => import('../../Lab/components/Lab'));
const HmFooter = React.lazy(() => import('./Footer'));

const App: React.VFC = () => (
  <div className={styles.hmApp}>
    <BrowserRouter>
      <HmLazyComponent>
        <Switch>
          <Route exact path={Paths.appRootPath} component={HmHome} />
          <Route exact path={Paths.signInPath} component={HmSignIn} />
          <Route exact path={Paths.labPath} component={HmLab} />
          <Redirect to={Paths.appRootPath} />
        </Switch>
      </HmLazyComponent>
    </BrowserRouter>

    <HmLazyComponent>
      <HmFooter />
    </HmLazyComponent>
  </div>
);

export default App;
