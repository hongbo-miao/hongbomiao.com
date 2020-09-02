import './shared/utils/initTracer';
import './index.css';
import React, { StrictMode } from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import HmApp from './App/components/App';
import * as serviceWorker from './shared/libs/serviceWorker';
import initSentry from './shared/utils/initSentry';
import store from './shared/utils/store';

initSentry();

ReactDOM.render(
  <StrictMode>
    <Provider store={store}>
      <HmApp />
    </Provider>
  </StrictMode>,
  document.getElementById('root')
);

serviceWorker.register();
