import 'core-js';
import React from 'react';
import ReactDOM from 'react-dom';
import 'bulma/css/bulma.css';
import 'normalize.css';

import * as serviceWorker from './shared/lib/serviceWorker';
import HmApp from './App/components/App';
import './index.css';

ReactDOM.render(<HmApp />, document.getElementById('root'));

serviceWorker.register();
