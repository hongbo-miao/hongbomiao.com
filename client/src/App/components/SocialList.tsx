import React from 'react';
import Website from '../types/Website.type';
import styles from './SocialList.module.css';

interface Props {
  websites: Website[];
}

const SocialList: React.FC<Props> = (props: Props) => {
  const { websites } = props;

  const socialListItems = websites.map((website) => {
    const { name, src, url } = website;

    return (
      <div key={name} className="level-item">
        <a href={url} target="_blank" rel="noopener noreferrer">
          <img className={styles.hmLogo} src={src} height="28" width="28" alt={name} />
        </a>
      </div>
    );
  });

  return <>{socialListItems}</>;
};

export default SocialList;
