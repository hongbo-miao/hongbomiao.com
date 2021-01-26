import { Request, Response } from 'express';
import logger from '../../log/utils/logger';

const uploadFile = (req: Request, res: Response): void => {
  logger.info(req.file, 'uploadFile');
  res.sendStatus(200);
};

export default uploadFile;
