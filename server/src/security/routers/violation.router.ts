import bodyParser from 'body-parser';
import { Router } from 'express';
import reportCSPViolation from '../controllers/reportCSPViolation';

const violationRouter = Router();
violationRouter.post('/report-csp-violation', bodyParser.json({ type: 'application/csp-report' }), reportCSPViolation);

export default violationRouter;
