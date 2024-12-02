import { useQuery } from '@tanstack/react-query';
import type { AxiosResponse } from 'axios';
import React, { Dispatch, SetStateAction } from 'react';

type Props = {
  children: React.JSX.Element;
  action: string;
  resource: string;
  getDecision: (action: string, resource: string) => Promise<AxiosResponse | null>;
  setData: Dispatch<SetStateAction<unknown>>;
};

function PreAuth(props: Props) {
  const { children, action, resource, getDecision, setData } = props;
  const { data } = useQuery([action, resource], () => getDecision(action, resource));

  React.useEffect(() => {
    setData(data?.data);
  }, [data, setData]);

  return children;
}

export default PreAuth;
