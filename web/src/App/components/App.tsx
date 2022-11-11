import React from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import HmHome from '../../Home/components/Home';
import HmLab from '../../Lab/components/Lab';
import HmSignIn from '../../SignIn/components/SignIn';
import Paths from '../../shared/utils/paths';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path={Paths.appRootPath} element={<HmHome />} />
        <Route path={Paths.signInPath} element={<HmSignIn />} />
        <Route path={Paths.labPath} element={<HmLab />} />
        <Route path="*" element={<Navigate to={Paths.appRootPath} replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
