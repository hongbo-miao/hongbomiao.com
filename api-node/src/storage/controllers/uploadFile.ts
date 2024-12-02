import { Request, Response } from 'express';
import logger from '../../log/utils/logger.js';

const uploadFile = (req: Request, res: Response): void => {
  logger.info(req.file, 'uploadFile');
  res.sendStatus(200);
};

export default uploadFile;
