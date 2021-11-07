import React from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import HmLazyComponent from '../../shared/components/LazyComponent';
import Paths from '../../shared/utils/paths';

const HmHome = React.lazy(() => import('../../Home/components/Home'));
const HmSignIn = React.lazy(() => import('../../SignIn/components/SignIn'));
const HmLab = React.lazy(() => import('../../Lab/components/Lab'));

const App: React.VFC = () => (
  <BrowserRouter>
    <HmLazyComponent>
      <Routes>
        <Route path={Paths.appRootPath} element={<HmHome />} />
        <Route path={Paths.signInPath} element={<HmSignIn />} />
        <Route path={Paths.labPath} element={<HmLab />} />
        <Route path="*" element={<Navigate to={Paths.appRootPath} />} />
      </Routes>
    </HmLazyComponent>
  </BrowserRouter>
);

export default App;
