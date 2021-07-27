import { AxiosResponse } from 'axios';
import React, { Dispatch, SetStateAction } from 'react';
import { useQuery } from 'react-query';

type Props = {
  children: JSX.Element;
  action: string;
  resourceType: string;
  getDecision: (action: string, resourceType: string) => Promise<AxiosResponse | null>;
  setData: Dispatch<SetStateAction<unknown>>;
};

const PreAuth: React.VFC<Props> = (props) => {
  const { children, action, resourceType, getDecision, setData } = props;
  const { data } = useQuery([action, resourceType], () => getDecision(action, resourceType));

  React.useEffect(() => {
    setData(data?.data);
  }, [data, setData]);

  return <>{children}</>;
};

export default PreAuth;
