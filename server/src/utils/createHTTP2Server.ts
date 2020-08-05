import fs from 'fs';
import path from 'path';
import { Application } from 'express';
import spdy, { Server } from 'spdy';

const createHTTP2Server = (app: Application): Server => {
  return spdy.createServer(
    {
      key: fs.readFileSync(path.join(__dirname, '../../private/ssl/hongbomiao.key')),
      cert: fs.readFileSync(path.join(__dirname, '../../private/ssl/hongbomiao.crt')),
    },
    app
  );
};

export default createHTTP2Server;
