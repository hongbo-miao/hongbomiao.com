import listEndpoints from 'express-list-endpoints';
import apiRouter from '../src/shared/routers/apiRouter';

// eslint-disable-next-line no-console
console.table(listEndpoints(apiRouter));
