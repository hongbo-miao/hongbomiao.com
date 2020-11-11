import bodyParser from 'body-parser';
import timeout from 'connect-timeout';
import csrf from 'csurf';
import { Router } from 'express';
import multer from 'multer';
import violationRouter from '../../security/routers/violation.router';
import uploadFile from '../../storage/controllers/uploadFile';

const csrfProtection = csrf({ cookie: true });
const parseForm = bodyParser.urlencoded({ extended: false });
const upload = multer();

const apiRouter = Router()
  .use('/violation', timeout('5s'), violationRouter)
  .use('/upload-file', parseForm, csrfProtection, upload.single('file'), uploadFile);

export default apiRouter;
