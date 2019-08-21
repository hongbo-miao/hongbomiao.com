import React from 'react';

import Websites from '../fixtures/websites';
import './Footer.css';


function Footer() {
  function renderLogos() {
    return Websites.map((website) => {
      const {
        name,
        src,
        url,
      } = website;

      return (
        <div key={name} className="level-item">
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
          >
            <img className="hm-logo" src={src} alt={name} />
          </a>
        </div>
      );
    });
  }

  return (
    <footer className="footer hm-footer">
      <div className="container hm-container">
        <nav className="level">
          <div className="level-left">
            {renderLogos()}
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
