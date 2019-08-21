import React, {
  lazy, Suspense,
} from 'react';

import HmLoading from '../../shared/components/Loading';
import Websites from '../fixtures/websites';
import './Footer.css';


const HmSocialList = lazy(() => import('./SocialList'));

function Footer() {
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
              <div className="hm-copyright">
                © 2019 H.M.
              </div>
            </div>
          </div>
        </nav>
      </div>
    </footer>
  );
}

export default Footer;
