import { Link } from '@tanstack/react-router';
import React from 'react';
import Paths from '../../shared/utils/paths';

function Menu() {
  return (
    <aside className="menu">
      <p className="menu-label">GENERAL</p>
      <ul className="menu-list">
        <li>
          <Link to={Paths.Lab.welcomePath} activeProps={{ className: 'is-active' }}>
            Welcome
          </Link>
        </li>
      </ul>
      <p className="menu-label">EXPERIMENTS</p>
      <ul className="menu-list">
        <li>
          <Link to={Paths.Lab.opaPath} activeProps={{ className: 'is-active' }}>
            OPA
          </Link>
        </li>
        <li>
          <Link to={Paths.Lab.opalPath} activeProps={{ className: 'is-active' }}>
            OPAL
          </Link>
        </li>
      </ul>
    </aside>
  );
}

export default Menu;
