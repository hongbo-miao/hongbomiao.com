import React from 'react';
import Website from '../types/Website';
import styles from './SocialList.module.css';

type Props = {
  websites: ReadonlyArray<Website>;
};

function SocialList(props: Props) {
  const { websites } = props;
  return websites.map((website) => {
    const { name, src, url } = website;
    return (
      <div key={name} className="level-item">
        <a href={url} target="_blank" rel="noopener noreferrer">
          <img className={styles.hmLogo} src={src} height="28" width="28" alt={name} />
        </a>
      </div>
    );
  });
}

export default SocialList;
