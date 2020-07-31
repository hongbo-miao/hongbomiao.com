import express from 'express';
import path from 'path';

import helmetMiddleware from './middlewares/helmet.middleware';
import corsMiddleware from './middlewares/cors.middleware';

const app: express.Application = express();
const port = 3001;

app.use(corsMiddleware);
app.use(helmetMiddleware);
app.use(express.static(path.join(__dirname, '../../client/build')));
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../../client/build/index.html'));
});

// eslint-disable-next-line no-console
app.listen(port, () => console.log(`Listening at http://localhost:${port}`));
