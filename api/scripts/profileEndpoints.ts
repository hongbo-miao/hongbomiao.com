import listEndpoints from 'express-list-endpoints';
import apiRouter from '../src/shared/routers/api.router';

// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
// eslint-disable-next-line no-console
console.table(listEndpoints(apiRouter));
