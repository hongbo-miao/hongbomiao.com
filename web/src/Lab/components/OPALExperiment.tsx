/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable no-console */

import React from 'react';
import adoptDog from '../queries/adoptDog';
import getDog from '../queries/getDog';
import getOPALDecision from '../queries/getOPALDecision';
import styles from './OPALExperiment.module.css';
import HmPreAuth from './PreAuth';

function OPALExperiment() {
  const [readDogData, setReadDogData] = React.useState<any>(undefined);
  const [adoptDogData, setAdoptDogData] = React.useState<any>(undefined);

  const onReadDog = async () => {
    const res = await getDog('1');
    console.log('dog', res?.data?.data?.dog);
  };

  const onAdoptDog = async () => {
    const res = await adoptDog('1');
    console.log('adoptDog', res?.data?.data?.adoptDog);
  };

  return (
    <div className={styles.hmOPALExperiment}>
      <div className={`container is-max-desktop ${styles.hmContainer}`}>
        <div className="buttons">
          <HmPreAuth action="read" resource="dog" getDecision={getOPALDecision} setData={setReadDogData}>
            <button
              className="button is-primary"
              type="button"
              disabled={!readDogData?.data?.opal?.decision}
              onClick={onReadDog}
            >
              Read Dog
            </button>
          </HmPreAuth>
          <HmPreAuth action="adopt" resource="dog" getDecision={getOPALDecision} setData={setAdoptDogData}>
            <button
              className="button is-primary"
              type="button"
              disabled={!adoptDogData?.data?.opal?.decision}
              onClick={onAdoptDog}
            >
              Adopt Dog
            </button>
          </HmPreAuth>
        </div>
      </div>
    </div>
  );
}

export default OPALExperiment;
