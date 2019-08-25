import React from 'react';

import Config from '../../config';
import './Copyright.css';


function Copyright(props) {
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
}

export default Copyright;
