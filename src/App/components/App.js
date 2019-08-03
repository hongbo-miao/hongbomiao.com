import React, {
  lazy,
  Suspense,
} from 'react';
import {
  Redirect,
  Route,
  Switch,
} from 'react-router-dom';

import Paths from '../../shared/utils/paths';
import HmLoading from '../../shared/components/Loading';
import './App.css';


const HmFooter = lazy(() => import('./Footer'));
const HmHome = lazy(() => import('../../Home/components/Home'));


function App() {
  return (
    <div className="hm-app">
      <Suspense fallback={HmLoading}>
        <Switch>
          <Route
            exact
            path={Paths.appRootPath}
            component={HmHome}
          />
          <Redirect
            to={Paths.appRootPath}
          />
        </Switch>
      </Suspense>

      <Suspense fallback={HmLoading}>
        <HmFooter />
      </Suspense>
    </div>
  );
}

export default App;
