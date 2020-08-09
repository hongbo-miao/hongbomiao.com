import { Router } from 'express';
import violationRouter from '../../security/routers/violation.router';

const apiRouter = Router();
apiRouter.post('/violation', violationRouter);

export default apiRouter;
