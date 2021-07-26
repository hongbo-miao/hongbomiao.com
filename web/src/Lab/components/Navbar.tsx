import React from 'react';
import useAuth from '../../auth/hooks/useAuth';

const Navbar: React.VFC = () => {
  const { signOut } = useAuth();
  return (
    <nav className="navbar" role="navigation" aria-label="main navigation">
      <div id="navbarBasicExample" className="navbar-menu">
        <div className="navbar-end">
          <div className="navbar-item">
            <div className="buttons">
              <button className="button is-primary" type="button" onClick={signOut}>
                Sign Out
              </button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
