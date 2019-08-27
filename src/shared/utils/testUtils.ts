import ReactDOM from 'react-dom';
import { ReactElement } from 'react';

const testComponent = (component: ReactElement): void => {
  const div = document.createElement('div');
  ReactDOM.render(component, div);
  ReactDOM.unmountComponentAtNode(div);
};

const TestUtils = {
  testComponent,
};

export default TestUtils;
