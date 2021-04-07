import React, { ReactNode, Suspense } from 'react';
import HmLoading from './Loading';

type Props = {
  children: ReactNode;
};

const LazyComponent: React.VFC<Props> = (props) => {
  const { children } = props;
  return <Suspense fallback={<HmLoading />}>{children}</Suspense>;
};

export default LazyComponent;
