import path from 'path';
import express from 'express';
import redirectSSL from 'redirect-ssl';

import corsMiddleware from './middlewares/cors.middleware';
import helmetMiddleware from './middlewares/helmet.middleware';
import rateLimitMiddleware from './middlewares/rate-limit.middleware';

const app = express();
const port = process.env.PORT || 3001;

app.use(corsMiddleware);
app.use(rateLimitMiddleware);
app.use(helmetMiddleware);
app.use(redirectSSL);
app.use(express.static(path.join(__dirname, '../../client/build')));
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../../client/build/index.html'));
});

// eslint-disable-next-line no-console
app.listen(port, () => console.log(`Listening at port ${port}`));
