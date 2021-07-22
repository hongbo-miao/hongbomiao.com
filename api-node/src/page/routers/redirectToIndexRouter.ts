import { Router } from 'express';
import redirectToIndexPage from '../controllers/redirectToIndexPage';

const redirectToIndexRouter = Router().get('/*', redirectToIndexPage);

export default redirectToIndexRouter;
