import React from 'react';
import getOPADecision from '../queries/getOPADecision';
import styles from './OPAExperiment.module.css';
import HmPreAuth from './PreAuth';

function OPAExperiment() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [readDogData, setReadDogData] = React.useState<any>(undefined);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [adoptDogData, setAdoptDogData] = React.useState<any>(undefined);

  return (
    <div className={styles.hmOPAExperiment}>
      <div className={`container is-max-desktop ${styles.hmContainer}`}>
        <div className="buttons">
          <HmPreAuth action="read" resource="dog" getDecision={getOPADecision} setData={setReadDogData}>
            <button className="button is-primary" type="button" disabled={!readDogData?.data?.opa?.decision}>
              Read Dog
            </button>
          </HmPreAuth>
          <HmPreAuth action="adopt" resource="dog" getDecision={getOPADecision} setData={setAdoptDogData}>
            <button className="button is-primary" type="button" disabled={!adoptDogData?.data?.opa?.decision}>
              Adopt Dog
            </button>
          </HmPreAuth>
        </div>
      </div>
    </div>
  );
}

export default OPAExperiment;
