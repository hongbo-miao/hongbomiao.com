import React from 'react';

import Config from '../../config';
import './Copyright.css';


interface Props {
  year: number;
}

const Copyright: React.FC<Props> = (props: Props) => {
  const {
    year,
  } = props;

  const copyright = `© ${year} H.M.`;

  return (
    <a
      className="hm-copyright"
      href={Config.githubUrl}
      target="_blank"
      rel="noopener noreferrer"
    >
      {copyright}
    </a>
  );
};

export default Copyright;
