import bodyParser from 'body-parser';
import { Router } from 'express';
import reportCSPViolation from '../controllers/violation/reportCSPViolation';

const apiRouter = Router();
apiRouter.post('/report-csp-violation', bodyParser.json({ type: 'application/csp-report' }), reportCSPViolation);

export default apiRouter;
