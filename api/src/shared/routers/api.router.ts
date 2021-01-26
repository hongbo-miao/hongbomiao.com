import bodyParser from 'body-parser';
import timeout from 'connect-timeout';
import csrf from 'csurf';
import { Router } from 'express';
import multer from 'multer';
import authMiddleware from '../../security/middlewares/auth.middleware';
import violationRouter from '../../security/routers/violation.router';
import uploadFile from '../../storage/controllers/uploadFile';

const csrfProtection = csrf({ cookie: { key: '__Host-csrf' } });
const parseForm = bodyParser.urlencoded({ extended: false });
const upload = multer({
  limits: { fileSize: 1e6 }, // 1MB
});

const apiRouter = Router()
  .use('/violation', timeout('5s'), violationRouter)
  .post('/upload-file', authMiddleware(), parseForm, csrfProtection, upload.single('file'), uploadFile);

export default apiRouter;
