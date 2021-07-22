import { Request, Response } from 'express';
import logger from '../../log/utils/logger';

const uploadFile = (req: Request, res: Response): void => {
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  logger.info(req.file, 'uploadFile');
  res.sendStatus(200);
};

export default uploadFile;
