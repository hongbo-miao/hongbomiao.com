import { Request, Response } from 'express';

const redirectToIndexPage = (req: Request, res: Response): void => {
  res.redirect('/');
};

export default redirectToIndexPage;
