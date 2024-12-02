import fs from 'fs';
import { Request, Response } from 'express';
import attachCSPNonce from '../../security/utils/attachCSPNonce.js';

const sendIndexPage = (req: Request, res: Response): void => {
  const { cspNonce } = res.locals;
  const html = fs.readFileSync('public/index.html', 'utf-8');
  res.send(attachCSPNonce(html, cspNonce));
};

export default sendIndexPage;
