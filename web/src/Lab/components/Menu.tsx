import React from 'react';
import { NavLink } from 'react-router-dom';
import Paths from '../../shared/utils/paths';

function Menu() {
  return (
    <aside className="menu">
      <p className="menu-label">GENERAL</p>
      <ul className="menu-list">
        <li>
          <NavLink end className={({ isActive }) => (isActive ? ' is-active' : '')} to={Paths.Lab.welcomePath}>
            Welcome
          </NavLink>
        </li>
      </ul>
      <p className="menu-label">EXPERIMENTS</p>
      <ul className="menu-list">
        <li>
          <NavLink className={({ isActive }) => (isActive ? ' is-active' : '')} to={Paths.Lab.opaPath}>
            OPA
          </NavLink>
        </li>
        <li>
          <NavLink className={({ isActive }) => (isActive ? ' is-active' : '')} to={Paths.Lab.opalPath}>
            OPAL
          </NavLink>
        </li>
      </ul>
    </aside>
  );
}

export default Menu;
