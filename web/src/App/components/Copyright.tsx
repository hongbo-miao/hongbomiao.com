import React from 'react';
import './Copyright.css';

type Props = {
  year: number;
};

function Copyright(props: Props) {
  const { year } = props;
  const copyright = `© ${year} H.M.`;
  return <div className="hm-copyright">{copyright}</div>;
}

export default Copyright;
