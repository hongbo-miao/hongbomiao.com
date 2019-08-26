import React from 'react';

import './SocialList.css';


interface Website {
  name: string;
  src: string;
  url: string;
}

export interface Props {
  websites: Array<Website>;
}

const SocialList: React.FC<Props> = (props: Props) => {
  const {
    websites,
  } = props;

  const socialListItems = websites.map((website) => {
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

  return (
    <>
      {socialListItems}
    </>
  );
};

export default SocialList;
