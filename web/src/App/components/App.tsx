import React from 'react';
import { BrowserRouter, Redirect, Route, Switch } from 'react-router-dom';
import HmLazyComponent from '../../shared/components/LazyComponent';
import Paths from '../../shared/utils/paths';

const HmHome = React.lazy(() => import('../../Home/components/Home'));
const HmSignIn = React.lazy(() => import('../../SignIn/components/SignIn'));
const HmLab = React.lazy(() => import('../../Lab/components/Lab'));

const App: React.VFC = () => (
  <BrowserRouter>
    <HmLazyComponent>
      <Switch>
        <Route exact path={Paths.appRootPath} component={HmHome} />
        <Route exact path={Paths.signInPath} component={HmSignIn} />
        <Route path={Paths.labPath} component={HmLab} />
        <Redirect to={Paths.appRootPath} />
      </Switch>
    </HmLazyComponent>
  </BrowserRouter>
);

export default App;
