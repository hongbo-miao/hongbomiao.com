import React from 'react';
import styles from './OPAExperiment.module.css';

const HmPreAuth = React.lazy(() => import('./PreAuth'));

const OPAExperiment: React.VFC = () => {
  const [isReadDogDisabled, setReadDogIsDisabled] = React.useState(false);
  const [isAdoptDogDisabled, setIsAdoptDogDisabled] = React.useState(false);

  return (
    <div className={styles.hmOPAExperiment}>
      <div className={`container is-max-desktop ${styles.hmContainer}`}>
        <div className="buttons">
          <HmPreAuth action="read" resourceType="dog" setIsDisabled={setReadDogIsDisabled}>
            <button className="button is-link" type="button" disabled={!isReadDogDisabled}>
              Read Dog
            </button>
          </HmPreAuth>
          <HmPreAuth action="adopt" resourceType="dog" setIsDisabled={setIsAdoptDogDisabled}>
            <button className="button is-link" type="button" disabled={!isAdoptDogDisabled}>
              Adopt Dog
            </button>
          </HmPreAuth>
        </div>
      </div>
    </div>
  );
};

export default OPAExperiment;
