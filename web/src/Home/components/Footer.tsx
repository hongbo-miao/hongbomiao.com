import React from 'react';
import HmCopyright from '../../App/components/Copyright';
import HmSocialList from '../../App/components/SocialList';
import WEBSITES from '../fixtures/WEBSITES';
import './Footer.css';

function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="bg-background hm-footer">
      <div className="max-w-4xl mx-auto px-6 hm-footer-container">
        <nav className="flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <HmSocialList websites={WEBSITES} />
          </div>
          <div className="flex items-center">
            <HmCopyright year={year} />
          </div>
        </nav>
      </div>
    </footer>
  );
}

export default Footer;
