import fs from 'fs';
import path from 'path';
import { Request, Response } from 'express';
import attachCSPNonce from '../../security/utils/attachCSPNonce';

const sendIndexPage = (req: Request, res: Response): void => {
  const { cspNonce } = res.locals;
  const html = fs.readFileSync(path.join(__dirname, '../../../../public/index.html'), 'utf-8');
  res.cookie('__Host-csrfToken', req.csrfToken(), {
    sameSite: 'strict',
    httpOnly: true,
    secure: true,
  });
  res.send(attachCSPNonce(html, cspNonce));
};

export default sendIndexPage;
