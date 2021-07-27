import React from 'react';
import getOPALDecision from '../queries/getOPALDecision';
import styles from './OPALExperiment.module.css';

const HmPreAuth = React.lazy(() => import('./PreAuth'));

const OPALExperiment: React.VFC = () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [readDogData, setReadDogData] = React.useState<any>(undefined);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [adoptDogData, setAdoptDogData] = React.useState<any>(undefined);

  return (
    <div className={styles.hmOPALExperiment}>
      <div className={`container is-max-desktop ${styles.hmContainer}`}>
        <div className="buttons">
          <HmPreAuth action="read" resourceType="dog" getDecision={getOPALDecision} setData={setReadDogData}>
            <button className="button is-link" type="button" disabled={!readDogData?.data?.opal?.decision}>
              Read Dog
            </button>
          </HmPreAuth>
          <HmPreAuth action="adopt" resourceType="dog" getDecision={getOPALDecision} setData={setAdoptDogData}>
            <button className="button is-link" type="button" disabled={!adoptDogData?.data?.opal?.decision}>
              Adopt Dog
            </button>
          </HmPreAuth>
        </div>
      </div>
    </div>
  );
};

export default OPALExperiment;
