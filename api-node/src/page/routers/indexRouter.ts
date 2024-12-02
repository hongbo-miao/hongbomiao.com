import { Router } from 'express';
import sendIndexPage from '../controllers/sendIndexPage.js';

const indexRouter = Router().get('/', sendIndexPage);

export default indexRouter;
