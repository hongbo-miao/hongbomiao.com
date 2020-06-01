import express from 'express';

const app: express.Application = express();
const port = 3001;

app.get('/', (req, res) => res.send('Hello World!'));

app.listen(port, () => console.log(`Listening at http://localhost:${port}`));
