import React, { StrictMode } from 'react';
import ReactDOM from 'react-dom';

import './index.css';
import * as serviceWorker from './shared/lib/serviceWorker';
import HmApp from './App/components/App';

ReactDOM.render(
  <StrictMode>
    <HmApp />
  </StrictMode>,
  document.getElementById('root')
);

// eslint-disable-next-line no-console
window.addEventListener('securitypolicyviolation', console.error.bind(console));

serviceWorker.register();
