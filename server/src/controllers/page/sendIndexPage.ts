import path from 'path';
import { Request, Response } from 'express';

const sendIndexPage = (req: Request, res: Response): void => {
  res.sendFile(path.join(__dirname, '../../client/build/index.html'));
};

export default sendIndexPage;
