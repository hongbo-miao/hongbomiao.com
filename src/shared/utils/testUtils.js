import ReactDOM from 'react-dom';


function testComponent(component) {
  const div = document.createElement('div');
  ReactDOM.render(component, div);
  ReactDOM.unmountComponentAtNode(div);
}

const TestUtils = {
  testComponent,
};

export default TestUtils;
