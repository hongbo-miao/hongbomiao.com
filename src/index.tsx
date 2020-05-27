import React from 'react';
import ReactDOM from 'react-dom';

import './index.css';
import * as serviceWorker from './shared/lib/serviceWorker';
import HmApp from './App/components/App';

ReactDOM.render(<HmApp />, document.getElementById('root'));

serviceWorker.register();
