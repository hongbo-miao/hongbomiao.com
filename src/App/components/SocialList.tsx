import React from 'react';

import Website from '../typings/website';
import './SocialList.css';

interface Props {
  websites: Website[];
}

const SocialList: React.FC<Props> = (props: Props) => {
  const { websites } = props;

  const socialListItems = websites.map(website => {
    const { name, src, url } = website;

    return (
      <div key={name} className="level-item hm-social-item">
        <a href={url} target="_blank" rel="noopener noreferrer">
          <img className="hm-logo" src={src} alt={name} />
        </a>
      </div>
    );
  });

  return <>{socialListItems}</>;
};

export default SocialList;
