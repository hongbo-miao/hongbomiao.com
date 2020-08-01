import { ReactElement } from 'react';
import ReactDOM from 'react-dom';

const testComponent = (component: ReactElement): void => {
  const div = document.createElement('div');
  ReactDOM.render(component, div);
  ReactDOM.unmountComponentAtNode(div);
};

const TestUtils = {
  testComponent,
};

export default TestUtils;
