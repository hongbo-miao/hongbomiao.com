import React from 'react';
import styles from './Copyright.module.css';

type Props = {
  year: number;
};

const Copyright: React.FC<Props> = (props) => {
  const { year } = props;
  const copyright = `© ${year} H.M.`;
  return <div className={styles.hmCopyright}>{copyright}</div>;
};

export default Copyright;
