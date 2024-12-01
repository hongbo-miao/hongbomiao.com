import { Router } from 'express';
import sendIndexPage from '../controllers/sendIndexPage';

const indexRouter = Router().get('/', sendIndexPage);

export default indexRouter;
