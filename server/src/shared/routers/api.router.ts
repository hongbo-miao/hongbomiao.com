import { Router } from 'express';
import violationRouter from '../../security/routers/violation.router';

const apiRouter = Router();
apiRouter.use('/violation', violationRouter);

export default apiRouter;
