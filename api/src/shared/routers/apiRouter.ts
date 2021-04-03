import timeout from 'connect-timeout';
import csrf from 'csurf';
import express, { Router } from 'express';
import multer from 'multer';
import authMiddleware from '../../security/middlewares/authMiddleware';
import violationRouter from '../../security/routers/violationRouter';
import uploadFile from '../../storage/controllers/uploadFile';

const csrfProtection = csrf({ cookie: { key: '__Host-csrf' } });
const parseForm = express.urlencoded({ extended: false });
const upload = multer({
  limits: { fileSize: 1e6 }, // 1MB
});

const apiRouter = Router()
  .use('/violation', timeout('5s'), violationRouter)
  .post('/upload-file', authMiddleware(), parseForm, csrfProtection, upload.single('file'), uploadFile);

export default apiRouter;
