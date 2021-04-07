import React from 'react';
import HmLazyComponent from '../../shared/components/LazyComponent';
import WEBSITES from '../fixtures/WEBSITES';
import styles from './Footer.module.css';

const HmCopyright = React.lazy(() => import('./Copyright'));
const HmSocialList = React.lazy(() => import('./SocialList'));

const Footer: React.VFC = () => {
  const year = new Date().getFullYear();

  return (
    <footer className={`footer ${styles.hmFooter}`}>
      <div className={`container is-max-desktop ${styles.hmContainer}`}>
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
