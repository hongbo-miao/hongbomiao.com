import React, { lazy } from 'react';

import WEBSITES from '../fixtures/websites';
import LazyComponent from '../../shared/components/LazyComponent';
import styles from './Footer.module.css';

const HmCopyright = lazy(() => import('./Copyright'));
const HmSocialList = lazy(() => import('./SocialList'));

const Footer: React.FC = () => {
  const year = new Date().getFullYear();

  return (
    <footer className={`footer ${styles.hmFooter}`}>
      <div className={`container ${styles.hmContainer}`}>
        <nav className="level">
          <div className="level-left">
            <LazyComponent>
              <HmSocialList websites={WEBSITES} />
            </LazyComponent>
          </div>
          <div className="level-right">
            <div className="level-item">
              <LazyComponent>
                <HmCopyright year={year} />
              </LazyComponent>
            </div>
          </div>
        </nav>
      </div>
    </footer>
  );
};

export default Footer;
