import React, {
  lazy, Suspense,
} from 'react';

import Config from '../../config';
import Websites from '../fixtures/websites';
import HmLoading from '../../shared/components/Loading';
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
              <a
                className="hm-copyright"
                href={Config.githubUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                © 2019 H.M.
              </a>
            </div>
          </div>
        </nav>
      </div>
    </footer>
  );
}

export default Footer;
