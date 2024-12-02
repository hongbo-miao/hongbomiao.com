import { Router } from 'express';
import redirectToIndexPage from '../controllers/redirectToIndexPage.js';

const redirectToIndexRouter = Router().get('/*', redirectToIndexPage);

export default redirectToIndexRouter;
