import path from 'path';
import { Request, Response } from 'express';

const sendIndexPage = (req: Request, res: Response): void => {
  res.sendFile(path.join(__dirname, '../dist/index.html'));
};

export default sendIndexPage;
