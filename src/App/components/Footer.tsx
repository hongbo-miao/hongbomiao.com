import React, { lazy, Suspense } from 'react';

import WEBSITES from '../fixtures/websites';
import HmLoading from '../../shared/components/Loading';
import './Footer.css';

const HmCopyright = lazy(() => import('./Copyright'));
const HmSocialList = lazy(() => import('./SocialList'));

const Footer: React.FC = () => {
  const year = new Date().getFullYear();

  return (
    <footer className="footer hm-footer">
      <div className="container hm-container">
        <nav className="level">
          <div className="level-left">
            <Suspense fallback={<HmLoading />}>
              <HmSocialList websites={WEBSITES} />
            </Suspense>
          </div>
          <div className="level-right">
            <div className="level-item">
              <Suspense fallback={<HmLoading />}>
                <HmCopyright year={year} />
              </Suspense>
            </div>
          </div>
        </nav>
      </div>
    </footer>
  );
};

export default Footer;
