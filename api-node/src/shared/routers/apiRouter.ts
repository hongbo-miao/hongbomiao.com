import timeout from 'connect-timeout';
import express, { Router } from 'express';
import multer from 'multer';
import authMiddleware from '../../security/middlewares/authMiddleware.js';
import violationRouter from '../../security/routers/violationRouter.js';
import uploadFile from '../../storage/controllers/uploadFile.js';

const parseForm = express.urlencoded({ extended: false });
const upload = multer({
  limits: { fileSize: 1e6 }, // 1MB
});

const apiRouter = Router()
  .use('/violation', timeout('5s'), violationRouter)
  .post('/upload-file', authMiddleware(), parseForm, upload.single('file'), uploadFile);

export default apiRouter;
