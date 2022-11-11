import React from 'react';
import HmCopyright from '../../App/components/Copyright';
import HmSocialList from '../../App/components/SocialList';
import WEBSITES from '../fixtures/WEBSITES';
import styles from './Footer.module.css';

function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className={`footer ${styles.hmFooter}`}>
      <div className={`container is-max-desktop ${styles.hmContainer}`}>
        <nav className="level">
          <div className="level-left">
            <HmSocialList websites={WEBSITES} />
          </div>
          <div className="level-right">
            <div className="level-item">
              <HmCopyright year={year} />
            </div>
          </div>
        </nav>
      </div>
    </footer>
  );
}

export default Footer;
