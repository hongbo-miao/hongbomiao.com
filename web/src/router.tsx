import { createRouter, createRoute, createRootRoute } from '@tanstack/react-router';
import React from 'react';
import HmHome from './Home/components/Home';
import HmLab from './Lab/components/Lab';
import HmSignIn from './SignIn/components/SignIn';
import Paths from './shared/utils/paths';
import HmApp from './App/components/App';
import HmLazyComponent from './shared/components/LazyComponent';

const HmOPAExperiment = React.lazy(() => import('./Lab/components/OPAExperiment'));
const HmOPALExperiment = React.lazy(() => import('./Lab/components/OPALExperiment'));
const HmWelcome = React.lazy(() => import('./Lab/components/Welcome'));

const rootRoute = createRootRoute({
  component: HmApp,
});

const homeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: Paths.appRootPath,
  component: HmHome,
});

const signInRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: Paths.signInPath,
  component: HmSignIn,
});

const labRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/lab',
  component: HmLab,
}).addChildren([
  createRoute({
    getParentRoute: () => labRoute,
    path: '/',
    component: () => <HmLazyComponent><HmWelcome /></HmLazyComponent>,
  }),
  createRoute({
    getParentRoute: () => labRoute,
    path: '/opa',
    component: () => <HmLazyComponent><HmOPAExperiment /></HmLazyComponent>,
  }),
  createRoute({
    getParentRoute: () => labRoute,
    path: '/opal',
    component: () => <HmLazyComponent><HmOPALExperiment /></HmLazyComponent>,
  }),
]);

const notFoundRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '*',
  component: () => <p>Not Found</p>,
});

const routeTree = rootRoute.addChildren([homeRoute, signInRoute, labRoute, notFoundRoute]);

export const router = createRouter({ routeTree });
