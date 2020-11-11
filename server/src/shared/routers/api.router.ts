import timeout from 'connect-timeout';
import { Router } from 'express';
import multer from 'multer';
import violationRouter from '../../security/routers/violation.router';
import uploadFile from '../../storage/controllers/uploadFile';

const upload = multer();

const apiRouter = Router();
apiRouter.use('/violation', timeout('5s'), violationRouter);
apiRouter.use('/upload-file', upload.single('file'), uploadFile);

export default apiRouter;
