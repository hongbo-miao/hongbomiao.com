import { AxiosResponse } from 'axios';
import React, { Dispatch, SetStateAction } from 'react';
import { useQuery } from 'react-query';

type Props = {
  children: JSX.Element;
  action: string;
  resourceType: string;
  getDecision: (action: string, resourceType: string) => Promise<AxiosResponse | null>;
  setIsDisabled: Dispatch<SetStateAction<boolean>>;
};

const PreAuth: React.VFC<Props> = (props) => {
  const { children, action, resourceType, getDecision, setIsDisabled } = props;
  const query = useQuery([action, resourceType], () => getDecision(action, resourceType));

  React.useEffect(() => {
    setIsDisabled(query?.data?.data?.data?.opa?.decision);
  }, [query]);

  return <>{children}</>;
};

export default PreAuth;
