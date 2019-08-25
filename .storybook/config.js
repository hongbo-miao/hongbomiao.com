import { configure } from '@storybook/react';
import requireContext from 'require-context.macro';
import 'bulma/css/bulma.css';
import 'normalize.css';

import '../src/index.css';


const req = requireContext('../src', true, /\.stories\.jsx$/);

function loadStories() {
  req.keys().forEach(filename => req(filename));
}

configure(loadStories, module);
