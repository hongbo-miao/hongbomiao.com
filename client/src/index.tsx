import React, { StrictMode } from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import HmApp from './App/components/App';
import * as serviceWorker from './shared/lib/serviceWorker';

ReactDOM.render(
  <StrictMode>
    <HmApp />
  </StrictMode>,
  document.getElementById('root')
);

serviceWorker.register();
