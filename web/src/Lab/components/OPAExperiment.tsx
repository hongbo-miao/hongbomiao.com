import React from 'react';

const HmPreAuth = React.lazy(() => import('./PreAuth'));

const OPAExperiment: React.VFC = () => {
  const [isReadDogDisabled, setReadDogIsDisabled] = React.useState(false);
  const [isAdoptDogDisabled, setIsAdoptDogDisabled] = React.useState(false);

  return (
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
  );
};

export default OPAExperiment;
