import React from 'react';
import getOPALDecision from '../queries/getOPALDecision';
import styles from './OPALExperiment.module.css';

const HmPreAuth = React.lazy(() => import('./PreAuth'));

const OPALExperiment: React.VFC = () => {
  const [isReadDogDisabled, setReadDogIsDisabled] = React.useState(false);
  const [isAdoptDogDisabled, setIsAdoptDogDisabled] = React.useState(false);

  return (
    <div className={styles.hmOPALExperiment}>
      <div className={`container is-max-desktop ${styles.hmContainer}`}>
        <div className="buttons">
          <HmPreAuth
            action="read"
            resourceType="dog"
            getDecision={getOPALDecision}
            setIsDisabled={setReadDogIsDisabled}
          >
            <button className="button is-link" type="button" disabled={!isReadDogDisabled}>
              Read Dog
            </button>
          </HmPreAuth>
          <HmPreAuth
            action="adopt"
            resourceType="dog"
            getDecision={getOPALDecision}
            setIsDisabled={setIsAdoptDogDisabled}
          >
            <button className="button is-link" type="button" disabled={!isAdoptDogDisabled}>
              Adopt Dog
            </button>
          </HmPreAuth>
        </div>
      </div>
    </div>
  );
};

export default OPALExperiment;
