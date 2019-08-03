import React from 'react';

import hatImage from '../images/hat.png';
import magicImage from '../images/magic.png';
import './Home.css';


function Home() {
  return (
    <div className="container hm-container">
      <h1 className="title">HONGBO MIAO</h1>
      <div className="hm-content">
        <img className="hm-emoji" src={hatImage} alt="Magical Hat" />
        <div className="hm-text">Making magic happen</div>
        <img className="hm-emoji" src={magicImage} alt="Magic" />
      </div>
    </div>
  );
}

export default Home;
