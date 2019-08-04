import 'core-js';
import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter } from 'react-router-dom';
import 'bulma/css/bulma.css';
import 'normalize.css';

import * as serviceWorker from './service-worker';
import HmApp from './App/components/App';
import './index.css';


ReactDOM.render(
  (
    <BrowserRouter>
      <HmApp />
    </BrowserRouter>
  ),
  document.getElementById('root'),
);

serviceWorker.register();
