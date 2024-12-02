import listEndpoints from 'express-list-endpoints';
import apiRouter from '../src/shared/routers/apiRouter';

console.table(listEndpoints(apiRouter));
