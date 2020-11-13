import csrf from 'csurf';
import { Router } from 'express';
import sendIndexPage from '../controllers/sendIndexPage';

const indexRouter = Router().get(
  '/',
  csrf({
    cookie: {
      httpOnly: true,
      sameSite: 'strict',
      secure: true,
    },
  }),
  sendIndexPage
);

export default indexRouter;
