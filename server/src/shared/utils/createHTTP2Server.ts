import fs from 'fs';
import path from 'path';
import express from 'express';
import spdy from 'spdy';

const createHTTP2Server = (app: express.Application): spdy.Server => {
  return spdy.createServer(
    {
      key: fs.readFileSync(path.join(__dirname, '../../../../private/ssl/hongbomiao.key')),
      cert: fs.readFileSync(path.join(__dirname, '../../../../private/ssl/hongbomiao.crt')),
    },
    app
  );
};

export default createHTTP2Server;
