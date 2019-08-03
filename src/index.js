import React, {
  Suspense,
} from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter, Route, Switch } from 'react-router-dom';

import * as serviceWorker from './serviceWorker';
import Paths from './shared/utils/paths';
import HmHome from './Home/components/Home';
import HmLoading from './shared/components/Loading';
import './index.css';


ReactDOM.render(
  (
    <BrowserRouter>
      <Suspense fallback={<HmLoading />}>
        <Switch>
          <Route exact path={Paths.AppRootPath} component={HmHome} />} />
        </Switch>
      </Suspense>
    </BrowserRouter>
  ),
  document.getElementById('root'),
);

serviceWorker.unregister();
