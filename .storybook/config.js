import 'bulma/css/bulma.css';
import requireContext from 'require-context.macro';
import { configure } from '@storybook/react';

import '../src/index.css';

const req = requireContext('../src', true, /\.stories\.tsx$/);

const loadStories = () => {
  req.keys().forEach(req);
};

configure(loadStories, module);
