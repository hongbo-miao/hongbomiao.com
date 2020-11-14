import csrf from 'csurf';
import { Router } from 'express';
import sendIndexPage from '../controllers/sendIndexPage';

const indexRouter = Router().get(
  '/',
  csrf({
    cookie: {
      key: '__Host-csrf',
      sameSite: 'strict',
      httpOnly: true,
      secure: true,
    },
  }),
  sendIndexPage
);

export default indexRouter;
