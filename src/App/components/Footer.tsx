import React, { lazy } from 'react';

import HmLazyComponent from '../../shared/components/LazyComponent';
import styles from './Footer.module.css';
import WEBSITES from '../fixtures/websites';

const HmCopyright = lazy(() => import('./Copyright'));
const HmSocialList = lazy(() => import('./SocialList'));

const Footer: React.FC = () => {
  const year = new Date().getFullYear();

  return (
    <footer className={`footer ${styles.hmFooter}`}>
      <div className={`container ${styles.hmContainer}`}>
        <nav className="level">
          <div className="level-left">
            <HmLazyComponent>
              <HmSocialList websites={WEBSITES} />
            </HmLazyComponent>
          </div>
          <div className="level-right">
            <div className="level-item">
              <HmLazyComponent>
                <HmCopyright year={year} />
              </HmLazyComponent>
            </div>
          </div>
        </nav>
      </div>
    </footer>
  );
};

export default Footer;
