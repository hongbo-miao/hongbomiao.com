import cors from 'cors';
import express from 'express';
import path from 'path';

import helmetMiddleware from './middlewares/helmet.middleware';

const app: express.Application = express();
const port = 3001;

app.use(cors());
app.use(helmetMiddleware);
app.use(express.static(path.join(__dirname, '../../client/build')));
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../../client/build/index.html'));
});

app.listen(port, () => console.log(`Listening at http://localhost:${port}`));
