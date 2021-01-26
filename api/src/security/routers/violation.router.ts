import bodyParser from 'body-parser';
import { Router } from 'express';
import reportCSPViolation from '../controllers/reportCSPViolation';
import reportTo from '../controllers/reportTo';

const violationRouter = Router()
  .post('/report-csp-violation', bodyParser.json({ type: 'application/csp-report' }), reportCSPViolation)
  .post('/report-to', bodyParser.json({ type: 'application/reports+json' }), reportTo);

export default violationRouter;
