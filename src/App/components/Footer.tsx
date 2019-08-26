import React, {
  lazy, Suspense,
} from 'react';

import Websites from '../fixtures/websites';
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
              <HmSocialList
                websites={Websites}
              />
            </Suspense>
          </div>
          <div className="level-right">
            <div className="level-item">
              <Suspense fallback={<HmLoading />}>
                <HmCopyright
                  year={year}
                />
              </Suspense>
            </div>
          </div>
        </nav>
      </div>
    </footer>
  );
};

export default Footer;
