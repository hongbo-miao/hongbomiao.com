import React from 'react';
import { NavLink } from 'react-router-dom';
import Paths from '../../shared/utils/paths';

const Menu: React.VFC = () => {
  return (
    <aside className="menu">
      <p className="menu-label">GENERAL</p>
      <ul className="menu-list">
        <li>
          <NavLink exact activeClassName="is-active" to={Paths.welcomePath}>
            Welcome
          </NavLink>
        </li>
      </ul>
      <p className="menu-label">EXPERIMENTS</p>
      <ul className="menu-list">
        <li>
          <NavLink activeClassName="is-active" to={Paths.opaPath}>
            OPA
          </NavLink>
        </li>
        <li>
          <NavLink activeClassName="is-active" to={Paths.opalPath}>
            OPAL
          </NavLink>
        </li>
      </ul>
    </aside>
  );
};

export default Menu;
