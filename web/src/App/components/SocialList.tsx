import React from 'react';
import Website from '../types/Website';
import styles from './SocialList.module.css';

type Props = {
  websites: ReadonlyArray<Website>;
};

const SocialList: React.FC<Props> = (props) => {
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
