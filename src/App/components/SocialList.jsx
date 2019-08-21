import React from 'react';

import './SocialList.css';


function SocialList(props) {
  const { websites } = props;

  return websites.map((website) => {
    const {
      name,
      src,
      url,
    } = website;

    return (
      <div key={name} className="level-item hm-social-item">
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

export default SocialList;
