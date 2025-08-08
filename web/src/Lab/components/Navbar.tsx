import { Link } from '@tanstack/react-router';
import React from 'react';
import useAuth from '../../auth/hooks/useAuth';
import useMe from '../../auth/hooks/useMe';
import Paths from '../../shared/utils/paths';

function Navbar() {
  const { signOut } = useAuth();
  const { me } = useMe();

  return (
    <nav className="navbar" role="navigation" aria-label="main navigation">
      <div id="navbarBasicExample" className="navbar-menu">
        <div className="navbar-end">
          <div className="navbar-item has-dropdown is-hoverable">
            <span className="navbar-link">{me?.name}</span>

            <div className="navbar-dropdown">
              <Link className="navbar-item" to={Paths.signInPath} onClick={signOut}>
                Sign Out
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
